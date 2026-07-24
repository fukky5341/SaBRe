## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 7200 seconds
Split limit: 100
Threshold: 59.3643246892


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586)
1: (-48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823)
2: (-41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995)
3: (-46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655)
4: (-53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318)
5: (-46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136)
6: (-42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388)
7: (-54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980)
8: (-56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498)
9: (-44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543)
10: (-65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228)
11: (-62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211)
12: (-50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959)
13: (-64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359)
14: (-104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220)
15: (-54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428)
16: (-68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681)
17: (-110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952)
18: (-58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113)
19: (-46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579)
20: (-40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157)
21: (-61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965)
22: (-67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385)
23: (-45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168)
24: (-60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080)
25: (-47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282)
26: (-66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205)
27: (-58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999)
28: (-44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950)
29: (-76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410)
30: (-56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116)
31: (-61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119)
32: (-48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451)
33: (-71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263)
34: (-59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855)
35: (-66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189)
36: (-58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920)
37: (-76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633)
38: (-71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031)
39: (-88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823)
40: (-59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338)
41: (-43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987)
42: (-32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.99 + 121.52 = 124.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -59.3940217, upper bound: 59.3940217

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1688

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3689607, upper bound: 59.3855864
time: 207.31 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3855864
time: 143.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 350.74 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 350.74
Output dim: 6, lower bound: -59.3689607, upper bound: 59.3855864
IS_B2, status: Status.UNKNOWN, split count: 1, time: 350.74
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3855864

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -85.1350250, 35.1761169, -85.0557861, 35.1168404, -120.2518616, 120.2319031
1: -48.2002487, 33.1245422, -48.1317596, 33.0766792, -81.2769318, 81.2563019
2: -41.0963135, 28.5706215, -40.9579506, 28.5468254, -69.6431427, 69.5285645
3: -46.4896240, 38.3929749, -46.3437271, 38.3541679, -84.8437881, 84.7367020
4: -53.5853081, 36.7136612, -53.4221992, 36.6823883, -90.2676926, 90.1358566
5: -46.2136459, 40.4553604, -46.0795021, 40.4159698, -86.6296158, 86.5348663
6: -42.0037689, 40.9428596, -41.9551125, 40.8185425, -82.8223114, 82.8979721
7: -54.5564499, 41.0941315, -54.4617996, 41.0618553, -95.6183014, 95.5559311
8: -56.7665939, 48.0776215, -56.5978088, 48.0389938, -104.8055878, 104.6754303
9: -44.0496140, 41.3045883, -44.0178146, 41.2047958, -85.2544098, 85.3224030
10: -65.4124451, 52.9047661, -65.3575897, 52.6700363, -118.0824814, 118.2623596
11: -62.5357513, 40.8366394, -62.4875488, 40.6713791, -103.2071304, 103.3241882
12: -50.4983406, 49.8276367, -50.4570618, 49.5307198, -100.0290604, 100.2846985
13: -64.4966431, 55.0090981, -64.4455948, 54.8973923, -119.3940353, 119.4546814
14: -104.2860641, 24.1039753, -104.2138062, 23.9228630, -128.2089233, 128.3177795
15: -54.4514046, 33.0844536, -54.3469353, 33.0368729, -87.4882812, 87.4313889
16: -68.2536621, 46.4657974, -68.1854324, 46.3308372, -114.5844879, 114.6512222
17: -110.0348511, 44.0577850, -109.9727783, 43.7467346, -153.7815857, 154.0305481
18: -58.7931938, 46.7226028, -58.7369919, 46.6116638, -105.4048615, 105.4595947
19: -46.7875214, 22.8095207, -46.7444305, 22.7391167, -69.5266342, 69.5539474
20: -40.6418610, 32.3330650, -40.5706558, 32.2950745, -72.9369354, 72.9037170
21: -61.4433174, 29.6637516, -61.3912277, 29.5678577, -91.0111771, 91.0549774
22: -67.8280487, 26.6438866, -67.7601166, 26.5686512, -94.3966980, 94.4040070
23: -45.6872635, 33.9522781, -45.6448822, 33.8659515, -79.5532150, 79.5971527
24: -60.3409920, 36.3404236, -60.2502098, 36.3209190, -96.6619110, 96.5906372
25: -47.1533279, 34.1296082, -47.1064529, 34.0778427, -81.2311707, 81.2360535
26: -66.6285095, 52.3272476, -66.5576477, 52.1978455, -118.8263550, 118.8848801
27: -58.8898125, 38.1182632, -58.7695312, 38.0973053, -96.9871216, 96.8877869
28: -43.9986153, 37.2873077, -43.9489479, 37.2555695, -81.2541809, 81.2362518
29: -76.0125580, 29.6271839, -75.9521484, 29.5229626, -105.5355225, 105.5793304
30: -56.4097023, 43.2578049, -56.3591499, 43.1980286, -99.6077271, 99.6169510
31: -61.5804710, 31.8053665, -61.5189590, 31.7429466, -93.3234177, 93.3243256
32: -48.2581406, 38.8870621, -48.2176285, 38.7799911, -87.0381317, 87.1046906
33: -71.8496552, 53.7435379, -71.7478638, 53.7028770, -125.5525360, 125.4914017
34: -59.1620674, 45.3361168, -59.0820732, 45.2831039, -104.4451752, 104.4181900
35: -66.6735382, 46.9312325, -66.5796204, 46.9040680, -113.5776062, 113.5108490
36: -58.9612846, 48.6878815, -58.8984680, 48.6482773, -107.6095581, 107.5863495
37: -76.3385468, 48.9140053, -76.2586060, 48.7916870, -125.1302338, 125.1726074
38: -70.9817886, 57.3369026, -70.8898468, 57.2528114, -128.2346039, 128.2267456
39: -88.3511658, 51.0265503, -88.2587891, 50.9949722, -139.3461304, 139.2853394
40: -59.1010399, 46.7673721, -58.9897423, 46.7318382, -105.8328781, 105.7571106
41: -43.7550735, 42.2029152, -43.7025871, 42.0897446, -85.8448181, 85.9055023
42: -32.8425293, 40.7548065, -32.8076324, 40.5895462, -73.4320755, 73.5624390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=196, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3059510, upper bound: 59.2985562
time: 113.66 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3059510, upper bound: 59.3734267
time: 1358.00 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -85.1691742, 35.1968346, -85.1973267, 35.1975021, -120.3666687, 120.3941650
1: -48.2304459, 33.1426430, -48.2348557, 33.1464157, -81.3768616, 81.3775024
2: -41.1587372, 28.5801849, -41.1587524, 28.6137924, -69.7725296, 69.7389374
3: -46.5555458, 38.4086494, -46.5562668, 38.4410439, -84.9965897, 84.9649124
4: -53.6586418, 36.7268829, -53.6618614, 36.7826080, -90.4412460, 90.3887482
5: -46.2745209, 40.4720764, -46.2766953, 40.5021744, -86.7766953, 86.7487640
6: -42.0245323, 40.9981232, -42.0693817, 40.9983139, -83.0228424, 83.0675049
7: -54.5976448, 41.1056519, -54.6044273, 41.1107826, -95.7084198, 95.7100830
8: -56.8423309, 48.0932198, -56.8457413, 48.1645775, -105.0069122, 104.9389648
9: -44.0632782, 41.3491211, -44.0753555, 41.3639183, -85.4272003, 85.4244766
10: -65.4360352, 53.0107117, -65.5016479, 53.0154266, -118.4514618, 118.5123444
11: -62.5547256, 40.9105225, -62.6081085, 40.9072456, -103.4619751, 103.5186310
12: -50.5154419, 49.9612007, -50.6389999, 49.9549942, -100.4704285, 100.6002045
13: -64.5152130, 55.0574379, -64.5175781, 55.0746231, -119.5898361, 119.5750122
14: -104.3159485, 24.1862316, -104.3607559, 24.1830635, -128.4990082, 128.5469818
15: -54.4792786, 33.1043129, -54.4619102, 33.1384811, -87.6177444, 87.5662231
16: -68.2825851, 46.5181274, -68.3243561, 46.5032387, -114.7858276, 114.8424835
17: -110.0610199, 44.1965561, -110.2187576, 44.1904869, -154.2515106, 154.4153137
18: -58.8153687, 46.7725029, -58.8488655, 46.7747078, -105.5900726, 105.6213684
19: -46.8061333, 22.8409481, -46.8487396, 22.8455601, -69.6516876, 69.6896820
20: -40.6724548, 32.3489914, -40.6906471, 32.3519516, -73.0244064, 73.0396423
21: -61.4657249, 29.7067451, -61.4924316, 29.7043991, -91.1701202, 91.1991730
22: -67.8503647, 26.6760216, -67.8576889, 26.6872692, -94.5376358, 94.5337067
23: -45.7056427, 33.9906044, -45.7295990, 33.9906006, -79.6962433, 79.7201996
24: -60.3793221, 36.3485947, -60.3925285, 36.3594780, -96.7387924, 96.7411194
25: -47.1720963, 34.1515656, -47.1896439, 34.1569977, -81.3290939, 81.3412094
26: -66.6501541, 52.3833389, -66.6572418, 52.3841171, -119.0342560, 119.0405807
27: -58.9418526, 38.1264458, -58.9542732, 38.1567078, -97.0985565, 97.0807190
28: -44.0196228, 37.2999649, -44.0342712, 37.3059540, -81.3255768, 81.3342361
29: -76.0338898, 29.6737556, -76.0443115, 29.6763515, -105.7102432, 105.7180634
30: -56.4295349, 43.2769775, -56.4416199, 43.2774544, -99.7069855, 99.7185974
31: -61.6074028, 31.8330917, -61.6491089, 31.8347321, -93.4421310, 93.4822006
32: -48.2754021, 38.9328232, -48.3046036, 38.9271660, -87.2025681, 87.2374268
33: -71.8944168, 53.7584991, -71.9095001, 53.7708054, -125.6652222, 125.6679993
34: -59.1969223, 45.3545914, -59.2041779, 45.3616142, -104.5585327, 104.5587692
35: -66.7143402, 46.9421616, -66.7222900, 46.9675217, -113.6818619, 113.6644516
36: -58.9845123, 48.7050171, -58.9907837, 48.7150879, -107.6996002, 107.6958008
37: -76.3714676, 48.9694633, -76.3999939, 48.9723434, -125.3438034, 125.3694534
38: -71.0201263, 57.3746681, -71.0374756, 57.3860893, -128.4062195, 128.4121399
39: -88.3898010, 51.0386162, -88.4116364, 51.0547485, -139.4445496, 139.4502411
40: -59.1500168, 46.7819595, -59.1718636, 46.7867126, -105.9367294, 105.9538269
41: -43.7778130, 42.2530365, -43.8297615, 42.2509079, -86.0287170, 86.0827942
42: -32.8575287, 40.8282471, -32.9505463, 40.8240967, -73.6816254, 73.7787933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=196, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3059510, upper bound: 59.2994947
time: 165.37 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3059510, upper bound: 59.3734267
time: 103.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 270.97 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 270.97
Output dim: 6, lower bound: -59.3059510, upper bound: 59.2985562
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 270.97
Output dim: 6, lower bound: -59.3059510, upper bound: 59.3734267
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 270.97
Output dim: 6, lower bound: -59.3059510, upper bound: 59.2994947
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 270.97
Output dim: 6, lower bound: -59.3059510, upper bound: 59.3734267

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -85.1263733, 35.1625748, -85.0555191, 35.1164093, -120.2427673, 120.2180939
1: -48.1930389, 33.1176796, -48.1315346, 33.0764618, -81.2695007, 81.2492142
2: -41.0869484, 28.5659962, -40.9576607, 28.5466805, -69.6336288, 69.5236511
3: -46.4782104, 38.3859291, -46.3433533, 38.3539467, -84.8321533, 84.7292786
4: -53.5676613, 36.7088165, -53.4216537, 36.6822243, -90.2498856, 90.1304626
5: -46.2040024, 40.4490967, -46.0792046, 40.4157677, -86.6197662, 86.5283051
6: -41.9973679, 40.9300842, -41.9548874, 40.8180275, -82.8153992, 82.8849716
7: -54.5479813, 41.0838013, -54.4615250, 41.0615349, -95.6095123, 95.5453262
8: -56.7526169, 48.0708389, -56.5973473, 48.0387650, -104.7913742, 104.6681824
9: -44.0444107, 41.2950668, -44.0176468, 41.2044907, -85.2489014, 85.3127136
10: -65.4048462, 52.8790894, -65.3573456, 52.6692276, -118.0740738, 118.2364349
11: -62.5286751, 40.8202667, -62.4873009, 40.6708565, -103.1995316, 103.3075638
12: -50.4919891, 49.8049622, -50.4568443, 49.5300217, -100.0220108, 100.2618027
13: -64.4819641, 54.9978600, -64.4451294, 54.8970299, -119.3789825, 119.4429779
14: -104.2748718, 24.0893841, -104.2134552, 23.9224091, -128.1972809, 128.3028412
15: -54.4252014, 33.0783157, -54.3459930, 33.0366402, -87.4618378, 87.4243088
16: -68.2440186, 46.4473724, -68.1851120, 46.3302650, -114.5742645, 114.6324768
17: -110.0260162, 44.0334320, -109.9725037, 43.7459602, -153.7719727, 154.0059357
18: -58.7831879, 46.7110214, -58.7366791, 46.6112900, -105.3944778, 105.4476929
19: -46.7814484, 22.8027821, -46.7442360, 22.7388916, -69.5203400, 69.5470123
20: -40.6349106, 32.3283081, -40.5704117, 32.2949219, -72.9298325, 72.8987198
21: -61.4364777, 29.6551666, -61.3909988, 29.5675850, -91.0040512, 91.0461578
22: -67.8049164, 26.6367512, -67.7593842, 26.5684147, -94.3733292, 94.3961334
23: -45.6823044, 33.9423828, -45.6447296, 33.8655815, -79.5478821, 79.5871124
24: -60.3283577, 36.3370590, -60.2498169, 36.3208237, -96.6491852, 96.5868683
25: -47.1445389, 34.1213264, -47.1061783, 34.0775528, -81.2220917, 81.2275085
26: -66.6038513, 52.3135910, -66.5568542, 52.1974030, -118.8012543, 118.8704453
27: -58.8751945, 38.1133270, -58.7690544, 38.0971489, -96.9723434, 96.8823853
28: -43.9913940, 37.2834587, -43.9487228, 37.2554398, -81.2468262, 81.2321777
29: -75.9993210, 29.6198502, -75.9517441, 29.5227337, -105.5220566, 105.5715866
30: -56.4018440, 43.2366714, -56.3588715, 43.1972809, -99.5991211, 99.5955429
31: -61.5734444, 31.7963867, -61.5187378, 31.7426567, -93.3161011, 93.3151169
32: -48.2523422, 38.8748703, -48.2174301, 38.7796135, -87.0319519, 87.0923004
33: -71.8364716, 53.7386551, -71.7474442, 53.7027054, -125.5391693, 125.4860992
34: -59.1503525, 45.3233109, -59.0817070, 45.2827110, -104.4330521, 104.4050140
35: -66.6588898, 46.9272804, -66.5791397, 46.9039230, -113.5628128, 113.5064240
36: -58.9486198, 48.6837997, -58.8980751, 48.6481323, -107.5967407, 107.5818787
37: -76.3184509, 48.9047470, -76.2579803, 48.7913742, -125.1098251, 125.1627274
38: -70.9671707, 57.3238182, -70.8893890, 57.2523766, -128.2195435, 128.2132111
39: -88.3194733, 51.0221939, -88.2577972, 50.9948235, -139.3143005, 139.2799835
40: -59.0879745, 46.7627716, -58.9892998, 46.7316933, -105.8196716, 105.7520752
41: -43.7481003, 42.1929169, -43.7023468, 42.0894318, -85.8375320, 85.8952637
42: -32.8382721, 40.7382240, -32.8074913, 40.5890121, -73.4272842, 73.5457153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=196, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3260866
time: 153.36 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3734269
time: 116.35 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -85.1605072, 35.1833115, -85.1970367, 35.1970673, -120.3575745, 120.3803406
1: -48.2231445, 33.1357727, -48.2346153, 33.1461945, -81.3693314, 81.3703918
2: -41.1493301, 28.5755386, -41.1584587, 28.6136303, -69.7629547, 69.7339935
3: -46.5440826, 38.4015770, -46.5559082, 38.4408188, -84.9849014, 84.9574814
4: -53.6409149, 36.7220345, -53.6613121, 36.7824440, -90.4233551, 90.3833466
5: -46.2648010, 40.4658241, -46.2763977, 40.5019684, -86.7667694, 86.7422180
6: -42.0181046, 40.9853363, -42.0691681, 40.9978409, -83.0159454, 83.0545044
7: -54.5890732, 41.0953064, -54.6041603, 41.1104469, -95.6995239, 95.6994629
8: -56.8282928, 48.0863876, -56.8452950, 48.1643600, -104.9926529, 104.9316788
9: -44.0580444, 41.3395844, -44.0751877, 41.3636169, -85.4216537, 85.4147720
10: -65.4283752, 52.9849701, -65.5014038, 53.0146103, -118.4429855, 118.4863663
11: -62.5476913, 40.8941269, -62.6078568, 40.9067307, -103.4544220, 103.5019836
12: -50.5090790, 49.9384613, -50.6387939, 49.9542923, -100.4633713, 100.5772552
13: -64.5004272, 55.0459976, -64.5171280, 55.0742760, -119.5746918, 119.5631256
14: -104.3046951, 24.1715927, -104.3603973, 24.1825981, -128.4872894, 128.5319824
15: -54.4542465, 33.0981941, -54.4609756, 33.1382790, -87.5925140, 87.5591736
16: -68.2729340, 46.4996071, -68.3240509, 46.5026474, -114.7755661, 114.8236465
17: -110.0521774, 44.1721649, -110.2184753, 44.1896973, -154.2418518, 154.3906250
18: -58.8052483, 46.7608910, -58.8485413, 46.7743301, -105.5795746, 105.6094208
19: -46.8000107, 22.8341732, -46.8485413, 22.8453350, -69.6453400, 69.6827087
20: -40.6655960, 32.3442192, -40.6904106, 32.3517990, -73.0173950, 73.0346298
21: -61.4588623, 29.6981316, -61.4922180, 29.7041283, -91.1629868, 91.1903458
22: -67.8268433, 26.6688786, -67.8569183, 26.6870327, -94.5138779, 94.5257874
23: -45.7006645, 33.9807892, -45.7294426, 33.9902153, -79.6908798, 79.7102280
24: -60.3666420, 36.3452377, -60.3921204, 36.3593750, -96.7260132, 96.7373581
25: -47.1632805, 34.1435471, -47.1893692, 34.1567421, -81.3200226, 81.3329010
26: -66.6254349, 52.3695221, -66.6564636, 52.3836594, -119.0090942, 119.0259857
27: -58.9271774, 38.1214828, -58.9537964, 38.1565323, -97.0837097, 97.0752792
28: -44.0123825, 37.2960663, -44.0340424, 37.3058243, -81.3182068, 81.3301086
29: -76.0206146, 29.6662903, -76.0438843, 29.6761150, -105.6967239, 105.7101746
30: -56.4217033, 43.2558594, -56.4413452, 43.2767181, -99.6984253, 99.6972046
31: -61.6003380, 31.8241005, -61.6488800, 31.8344402, -93.4347610, 93.4729767
32: -48.2695999, 38.9205475, -48.3044090, 38.9267845, -87.1963806, 87.2249603
33: -71.8812256, 53.7536011, -71.9090805, 53.7706451, -125.6518631, 125.6626816
34: -59.1851692, 45.3415909, -59.2037888, 45.3612061, -104.5463715, 104.5453796
35: -66.6996613, 46.9381866, -66.7218246, 46.9673920, -113.6670532, 113.6600113
36: -58.9717865, 48.7009048, -58.9903717, 48.7149506, -107.6867371, 107.6912766
37: -76.3513107, 48.9601021, -76.3993530, 48.9720421, -125.3233337, 125.3594513
38: -71.0054474, 57.3615074, -71.0370331, 57.3856659, -128.3911133, 128.3985443
39: -88.3580170, 51.0342865, -88.4106598, 51.0546036, -139.4126282, 139.4449463
40: -59.1369591, 46.7773170, -59.1714363, 46.7865601, -105.9235153, 105.9487457
41: -43.7708168, 42.2429657, -43.8295326, 42.2505875, -86.0214081, 86.0724945
42: -32.8532639, 40.8116150, -32.9503975, 40.8235741, -73.6768341, 73.7620087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=196, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2994946, upper bound: 59.3267095
time: 105.20 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3734269
time: 112.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 220.08 seconds
IS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 220.08
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3260866
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 220.08
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3734269
IS_B2_A2_B1, status: Status.VERIFIED, split count: 3, time: 220.08
Output dim: 6, lower bound: -59.2994946, upper bound: 59.3267095
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 220.08
Output dim: 6, lower bound: -59.2779264, upper bound: 59.3734269

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -85.1263733, 35.1625748, -85.0471191, 35.1029396, -120.2293091, 120.2096939
1: -48.1930389, 33.1176796, -48.1245193, 33.0697517, -81.2627869, 81.2422028
2: -41.0869484, 28.5659962, -40.9486084, 28.5421371, -69.6290894, 69.5146027
3: -46.4782104, 38.3859291, -46.3322067, 38.3471031, -84.8253174, 84.7181396
4: -53.5676613, 36.7088165, -53.4046860, 36.6774788, -90.2451401, 90.1135025
5: -46.2040024, 40.4490967, -46.0699348, 40.4096298, -86.6136322, 86.5190277
6: -41.9973679, 40.9300842, -41.9484062, 40.8056107, -82.8029785, 82.8784866
7: -54.5479813, 41.0838013, -54.4531288, 41.0515518, -95.5995331, 95.5369263
8: -56.7526169, 48.0708389, -56.5836830, 48.0321045, -104.7847214, 104.6545258
9: -44.0444107, 41.2950668, -44.0126266, 41.1951370, -85.2395477, 85.3076935
10: -65.4048462, 52.8790894, -65.3498840, 52.6444054, -118.0492554, 118.2289734
11: -62.5286751, 40.8202667, -62.4796715, 40.6550713, -103.1837463, 103.2999420
12: -50.4919891, 49.8049622, -50.4505501, 49.5081787, -100.0001678, 100.2555008
13: -64.4819641, 54.9978600, -64.4307251, 54.8860207, -119.3679810, 119.4285812
14: -104.2748718, 24.0893841, -104.2025681, 23.9082985, -128.1831665, 128.2919464
15: -54.4252014, 33.0783157, -54.3206635, 33.0303040, -87.4554977, 87.3989792
16: -68.2440186, 46.4473724, -68.1756134, 46.3123550, -114.5563583, 114.6229858
17: -110.0260162, 44.0334320, -109.9639511, 43.7221718, -153.7481842, 153.9973755
18: -58.7831879, 46.7110214, -58.7272148, 46.5999413, -105.3831329, 105.4382324
19: -46.7814484, 22.8027821, -46.7383575, 22.7322865, -69.5137329, 69.5411377
20: -40.6349106, 32.3283081, -40.5632248, 32.2903137, -72.9252243, 72.8915329
21: -61.4364777, 29.6551666, -61.3842354, 29.5593033, -90.9957809, 91.0393982
22: -67.8049164, 26.6367512, -67.7370148, 26.5610218, -94.3659363, 94.3737640
23: -45.6823044, 33.9423828, -45.6399155, 33.8556557, -79.5379562, 79.5822983
24: -60.3283577, 36.3370590, -60.2376480, 36.3175049, -96.6458588, 96.5747070
25: -47.1445389, 34.1213264, -47.0976143, 34.0692749, -81.2138138, 81.2189331
26: -66.6038513, 52.3135910, -66.5329971, 52.1842804, -118.7881317, 118.8465729
27: -58.8751945, 38.1133270, -58.7547340, 38.0922775, -96.9674683, 96.8680573
28: -43.9913940, 37.2834587, -43.9417496, 37.2515755, -81.2429581, 81.2252045
29: -75.9993210, 29.6198502, -75.9389648, 29.5155621, -105.5148849, 105.5588150
30: -56.4018440, 43.2366714, -56.3508301, 43.1765938, -99.5784378, 99.5874939
31: -61.5734444, 31.7963867, -61.5118713, 31.7338657, -93.3073120, 93.3082581
32: -48.2523422, 38.8748703, -48.2115402, 38.7678528, -87.0201950, 87.0864105
33: -71.8364716, 53.7386551, -71.7346344, 53.6976776, -125.5341339, 125.4732895
34: -59.1503525, 45.3233109, -59.0703201, 45.2706871, -104.4210358, 104.3936310
35: -66.6588898, 46.9272804, -66.5649185, 46.8999138, -113.5588074, 113.4922028
36: -58.9486198, 48.6837997, -58.8857803, 48.6441422, -107.5927582, 107.5695801
37: -76.3184509, 48.9047470, -76.2385406, 48.7821884, -125.1006393, 125.1432877
38: -70.9671707, 57.3238182, -70.8751144, 57.2394943, -128.2066650, 128.1989288
39: -88.3194733, 51.0221939, -88.2270966, 50.9904556, -139.3099213, 139.2492828
40: -59.0879745, 46.7627716, -58.9762650, 46.7271042, -105.8150787, 105.7390366
41: -43.7481003, 42.1929169, -43.6954193, 42.0797577, -85.8278580, 85.8883362
42: -32.8382721, 40.7382240, -32.8031769, 40.5729675, -73.4112396, 73.5413971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2776245, upper bound: 59.2802832
time: 73.51 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3734267
time: 82.22 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -85.1605072, 35.1833115, -85.1886292, 35.1840210, -120.3445282, 120.3719406
1: -48.2231445, 33.1357727, -48.2275314, 33.1395531, -81.3626938, 81.3633041
2: -41.1493301, 28.5755386, -41.1493759, 28.6091022, -69.7584229, 69.7249146
3: -46.5440826, 38.4015770, -46.5448647, 38.4339485, -84.9780273, 84.9464417
4: -53.6409149, 36.7220345, -53.6442528, 36.7777100, -90.4186249, 90.3662872
5: -46.2648010, 40.4658241, -46.2670746, 40.4958878, -86.7606888, 86.7328949
6: -42.0181046, 40.9853363, -42.0629120, 40.9854736, -83.0035782, 83.0482483
7: -54.5890732, 41.0953064, -54.5959358, 41.1004410, -95.6895142, 95.6912384
8: -56.8282928, 48.0863876, -56.8317719, 48.1577454, -104.9860382, 104.9181519
9: -44.0580444, 41.3395844, -44.0700760, 41.3543663, -85.4124069, 85.4096603
10: -65.4283752, 52.9849701, -65.4940186, 52.9896088, -118.4179840, 118.4789886
11: -62.5476913, 40.8941269, -62.6010017, 40.8908615, -103.4385529, 103.4951324
12: -50.5090790, 49.9384613, -50.6326332, 49.9323273, -100.4414062, 100.5710754
13: -64.5004272, 55.0459976, -64.5028229, 55.0632248, -119.5636292, 119.5488205
14: -104.3046951, 24.1715927, -104.3495331, 24.1683998, -128.4730988, 128.5211182
15: -54.4542465, 33.0981941, -54.4358139, 33.1323128, -87.5865555, 87.5340118
16: -68.2729340, 46.4996071, -68.3146210, 46.4847412, -114.7576752, 114.8142242
17: -110.0521774, 44.1721649, -110.2099304, 44.1660652, -154.2182465, 154.3820953
18: -58.8052483, 46.7608910, -58.8388672, 46.7630920, -105.5683441, 105.5997391
19: -46.8000107, 22.8341732, -46.8426056, 22.8387871, -69.6387939, 69.6767731
20: -40.6655960, 32.3442192, -40.6836128, 32.3471451, -73.0127411, 73.0278320
21: -61.4588623, 29.6981316, -61.4855232, 29.6957874, -91.1546478, 91.1836548
22: -67.8268433, 26.6688786, -67.8338013, 26.6800747, -94.5069122, 94.5026779
23: -45.7006645, 33.9807892, -45.7245865, 33.9806137, -79.6812744, 79.7053680
24: -60.3666420, 36.3452377, -60.3797646, 36.3560753, -96.7227173, 96.7249985
25: -47.1632805, 34.1435471, -47.1807098, 34.1489067, -81.3121872, 81.3242493
26: -66.6254349, 52.3695221, -66.6325836, 52.3704071, -118.9958344, 119.0021057
27: -58.9271774, 38.1214828, -58.9394875, 38.1517220, -97.0788879, 97.0609741
28: -44.0123825, 37.2960663, -44.0269623, 37.3020973, -81.3144836, 81.3230286
29: -76.0206146, 29.6662903, -76.0309906, 29.6689377, -105.6895447, 105.6972656
30: -56.4217033, 43.2558594, -56.4336777, 43.2562256, -99.6779251, 99.6895294
31: -61.6003380, 31.8241005, -61.6419983, 31.8257332, -93.4260712, 93.4660950
32: -48.2695999, 38.9205475, -48.2987289, 38.9148483, -87.1844482, 87.2192764
33: -71.8812256, 53.7536011, -71.8962479, 53.7658730, -125.6470947, 125.6498489
34: -59.1851692, 45.3415909, -59.1923828, 45.3489037, -104.5340729, 104.5339737
35: -66.6996613, 46.9381866, -66.7075348, 46.9635544, -113.6632156, 113.6457214
36: -58.9717865, 48.7009048, -58.9780121, 48.7109222, -107.6827087, 107.6789093
37: -76.3513107, 48.9601021, -76.3798065, 48.9631119, -125.3144226, 125.3399048
38: -71.0054474, 57.3615074, -71.0228271, 57.3729362, -128.3783569, 128.3843384
39: -88.3580170, 51.0342865, -88.3798828, 51.0503654, -139.4083862, 139.4141693
40: -59.1369591, 46.7773170, -59.1587257, 46.7820816, -105.9190369, 105.9360428
41: -43.7708168, 42.2429657, -43.8226814, 42.2408524, -86.0116730, 86.0656433
42: -32.8532639, 40.8116150, -32.9462433, 40.8074799, -73.6607437, 73.7578583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3562079
time: 129.44 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3734267
time: 103.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 235.18 seconds
IS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 235.18
Output dim: 6, lower bound: -59.2776245, upper bound: 59.2802832
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 235.18
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3734267
IS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 235.18
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3562079
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 235.18
Output dim: 6, lower bound: -59.2776245, upper bound: 59.3734267

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.1886292, 35.1840210, -85.0471191, 35.1029396, -120.2915649, 120.2311401
1: -48.2275314, 33.1395531, -48.1245193, 33.0697517, -81.2972794, 81.2640686
2: -41.1493759, 28.6091022, -40.9486084, 28.5421371, -69.6915131, 69.5577087
3: -46.5448647, 38.4339485, -46.3322067, 38.3471031, -84.8919678, 84.7661438
4: -53.6442528, 36.7777100, -53.4046860, 36.6774788, -90.3217239, 90.1823959
5: -46.2670746, 40.4958878, -46.0699348, 40.4096298, -86.6767044, 86.5658112
6: -42.0629120, 40.9854736, -41.9484062, 40.8056107, -82.8685226, 82.9338684
7: -54.5959358, 41.1004410, -54.4531288, 41.0515518, -95.6474762, 95.5535736
8: -56.8317719, 48.1577454, -56.5836830, 48.0321045, -104.8638535, 104.7414246
9: -44.0700760, 41.3543663, -44.0126266, 41.1951370, -85.2652130, 85.3669891
10: -65.4940186, 52.9896088, -65.3498840, 52.6444054, -118.1384201, 118.3394928
11: -62.6010017, 40.8908615, -62.4796715, 40.6550713, -103.2560577, 103.3705292
12: -50.6326332, 49.9323273, -50.4505501, 49.5081787, -100.1407928, 100.3828735
13: -64.5028229, 55.0632248, -64.4307251, 54.8860207, -119.3888397, 119.4939346
14: -104.3495331, 24.1683998, -104.2025681, 23.9082985, -128.2578278, 128.3709717
15: -54.4358139, 33.1323128, -54.3206635, 33.0303040, -87.4661179, 87.4529724
16: -68.3146210, 46.4847412, -68.1756134, 46.3123550, -114.6269760, 114.6603546
17: -110.2099304, 44.1660652, -109.9639511, 43.7221718, -153.9320984, 154.1300201
18: -58.8388672, 46.7630920, -58.7272148, 46.5999413, -105.4387970, 105.4903030
19: -46.8426056, 22.8387871, -46.7383575, 22.7322865, -69.5748901, 69.5771484
20: -40.6836128, 32.3471451, -40.5632248, 32.2903137, -72.9739227, 72.9103622
21: -61.4855232, 29.6957874, -61.3842354, 29.5593033, -91.0448227, 91.0800247
22: -67.8338013, 26.6800747, -67.7370148, 26.5610218, -94.3948212, 94.4170837
23: -45.7245865, 33.9806137, -45.6399155, 33.8556557, -79.5802383, 79.6205292
24: -60.3797646, 36.3560753, -60.2376480, 36.3175049, -96.6972656, 96.5937195
25: -47.1807098, 34.1489067, -47.0976143, 34.0692749, -81.2499771, 81.2465210
26: -66.6325836, 52.3704071, -66.5329971, 52.1842804, -118.8168488, 118.9033966
27: -58.9394875, 38.1517220, -58.7547340, 38.0922775, -97.0317688, 96.9064484
28: -44.0269623, 37.3020973, -43.9417496, 37.2515755, -81.2785339, 81.2438507
29: -76.0309906, 29.6689377, -75.9389648, 29.5155621, -105.5465546, 105.6079025
30: -56.4336777, 43.2562256, -56.3508301, 43.1765938, -99.6102753, 99.6070557
31: -61.6419983, 31.8257332, -61.5118713, 31.7338657, -93.3758545, 93.3376007
32: -48.2987289, 38.9148483, -48.2115402, 38.7678528, -87.0665741, 87.1263885
33: -71.8962479, 53.7658730, -71.7346344, 53.6976776, -125.5939026, 125.5005035
34: -59.1923828, 45.3489037, -59.0703201, 45.2706871, -104.4630737, 104.4192200
35: -66.7075348, 46.9635544, -66.5649185, 46.8999138, -113.6074524, 113.5284729
36: -58.9780121, 48.7109222, -58.8857803, 48.6441422, -107.6221466, 107.5967026
37: -76.3798065, 48.9631119, -76.2385406, 48.7821884, -125.1619873, 125.2016525
38: -71.0228271, 57.3729362, -70.8751144, 57.2394943, -128.2622986, 128.2480469
39: -88.3798828, 51.0503654, -88.2270966, 50.9904556, -139.3703308, 139.2774506
40: -59.1587257, 46.7820816, -58.9762650, 46.7271042, -105.8858337, 105.7583466
41: -43.8226814, 42.2408524, -43.6954193, 42.0797577, -85.9024353, 85.9362717
42: -32.9462433, 40.8074799, -32.8031769, 40.5729675, -73.5192108, 73.6106567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1703

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2736998, upper bound: 59.3394265
time: 132.83 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3376783, upper bound: 59.3713604
time: 153.61 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.1886292, 35.1840210, -85.1886292, 35.1840210, -120.3726501, 120.3726501
1: -48.2275314, 33.1395531, -48.2275314, 33.1395531, -81.3670807, 81.3670807
2: -41.1493759, 28.6091022, -41.1493759, 28.6091022, -69.7584610, 69.7584686
3: -46.5448647, 38.4339485, -46.5448647, 38.4339485, -84.9788055, 84.9788055
4: -53.6442528, 36.7777100, -53.6442528, 36.7777100, -90.4219589, 90.4219666
5: -46.2670746, 40.4958878, -46.2670746, 40.4958878, -86.7629471, 86.7629547
6: -42.0629120, 40.9854736, -42.0629120, 40.9854736, -83.0483856, 83.0483856
7: -54.5959358, 41.1004410, -54.5959358, 41.1004410, -95.6963730, 95.6963806
8: -56.8317719, 48.1577454, -56.8317719, 48.1577454, -104.9895096, 104.9895172
9: -44.0700760, 41.3543663, -44.0700760, 41.3543663, -85.4244308, 85.4244385
10: -65.4940186, 52.9896088, -65.4940186, 52.9896088, -118.4836273, 118.4836197
11: -62.6010017, 40.8908615, -62.6010017, 40.8908615, -103.4918671, 103.4918671
12: -50.6326332, 49.9323273, -50.6326332, 49.9323273, -100.5649567, 100.5649414
13: -64.5028229, 55.0632248, -64.5028229, 55.0632248, -119.5660400, 119.5660477
14: -104.3495331, 24.1683998, -104.3495331, 24.1683998, -128.5179291, 128.5179291
15: -54.4358139, 33.1323128, -54.4358139, 33.1323128, -87.5681229, 87.5681229
16: -68.3146210, 46.4847412, -68.3146210, 46.4847412, -114.7993622, 114.7993622
17: -110.2099304, 44.1660652, -110.2099304, 44.1660652, -154.3759918, 154.3759918
18: -58.8388672, 46.7630920, -58.8388672, 46.7630920, -105.6019440, 105.6019516
19: -46.8426056, 22.8387871, -46.8426056, 22.8387871, -69.6813965, 69.6813965
20: -40.6836128, 32.3471451, -40.6836128, 32.3471451, -73.0307541, 73.0307541
21: -61.4855232, 29.6957874, -61.4855232, 29.6957874, -91.1813049, 91.1813126
22: -67.8338013, 26.6800747, -67.8338013, 26.6800747, -94.5138779, 94.5138702
23: -45.7245865, 33.9806137, -45.7245865, 33.9806137, -79.7052002, 79.7052002
24: -60.3797646, 36.3560753, -60.3797646, 36.3560753, -96.7358398, 96.7358398
25: -47.1807098, 34.1489067, -47.1807098, 34.1489067, -81.3296127, 81.3296051
26: -66.6325836, 52.3704071, -66.6325836, 52.3704071, -119.0029755, 119.0029755
27: -58.9394875, 38.1517220, -58.9394875, 38.1517220, -97.0912094, 97.0912094
28: -44.0269623, 37.3020973, -44.0269623, 37.3020973, -81.3290558, 81.3290558
29: -76.0309906, 29.6689377, -76.0309906, 29.6689377, -105.6999207, 105.6999207
30: -56.4336777, 43.2562256, -56.4336777, 43.2562256, -99.6899033, 99.6899033
31: -61.6419983, 31.8257332, -61.6419983, 31.8257332, -93.4677277, 93.4677277
32: -48.2987289, 38.9148483, -48.2987289, 38.9148483, -87.2135620, 87.2135696
33: -71.8962479, 53.7658730, -71.8962479, 53.7658730, -125.6621170, 125.6621170
34: -59.1923828, 45.3489037, -59.1923828, 45.3489037, -104.5412903, 104.5412903
35: -66.7075348, 46.9635544, -66.7075348, 46.9635544, -113.6710815, 113.6710739
36: -58.9780121, 48.7109222, -58.9780121, 48.7109222, -107.6889267, 107.6889267
37: -76.3798065, 48.9631119, -76.3798065, 48.9631119, -125.3429108, 125.3429184
38: -71.0228271, 57.3729362, -71.0228271, 57.3729362, -128.3957520, 128.3957520
39: -88.3798828, 51.0503654, -88.3798828, 51.0503654, -139.4302521, 139.4302521
40: -59.1587257, 46.7820816, -59.1587257, 46.7820816, -105.9408112, 105.9408112
41: -43.8226814, 42.2408524, -43.8226814, 42.2408524, -86.0635376, 86.0635376
42: -32.9462433, 40.8074799, -32.9462433, 40.8074799, -73.7537231, 73.7537231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1703

## Relational analysis of IS_B2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3345563, upper bound: 59.3394279
time: 125.30 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3376783, upper bound: 59.3713607
time: 149.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 276.99 seconds
IS_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 276.99
Output dim: 6, lower bound: -59.2736998, upper bound: 59.3394265
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 276.99
Output dim: 6, lower bound: -59.3376783, upper bound: 59.3713604
IS_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 276.99
Output dim: 6, lower bound: -59.3345563, upper bound: 59.3394279
IS_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 276.99
Output dim: 6, lower bound: -59.3376783, upper bound: 59.3713607

## BFS IS instance: IS_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -85.1830673, 35.1761932, -85.0456619, 35.1009064, -120.2839661, 120.2218552
1: -48.2246056, 33.1344604, -48.1237679, 33.0684242, -81.2930298, 81.2582245
2: -41.1408997, 28.6025009, -40.9464226, 28.5404015, -69.6812973, 69.5489197
3: -46.5412598, 38.4186935, -46.3312454, 38.3430901, -84.8843536, 84.7499313
4: -53.6280365, 36.7740898, -53.4003372, 36.6764870, -90.3045197, 90.1744232
5: -46.2641068, 40.4817085, -46.0691338, 40.4059105, -86.6700134, 86.5508423
6: -42.0587540, 40.9775848, -41.9472961, 40.8035355, -82.8622894, 82.9248810
7: -54.5928192, 41.0919991, -54.4523354, 41.0493393, -95.6421509, 95.5443344
8: -56.8120613, 48.1521454, -56.5779343, 48.0306091, -104.8426590, 104.7300720
9: -44.0672913, 41.3480873, -44.0118561, 41.1935043, -85.2607880, 85.3599396
10: -65.4886017, 52.9772530, -65.3483734, 52.6411781, -118.1297684, 118.3256149
11: -62.5968552, 40.8829880, -62.4785461, 40.6529846, -103.2498398, 103.3615341
12: -50.6276131, 49.9165878, -50.4492073, 49.5040855, -100.1316986, 100.3657837
13: -64.4948044, 55.0509262, -64.4286041, 54.8827400, -119.3775406, 119.4795227
14: -104.3403778, 24.1553955, -104.2001495, 23.9046574, -128.2450256, 128.3555450
15: -54.4290962, 33.1288109, -54.3184128, 33.0293503, -87.4584427, 87.4472198
16: -68.3093491, 46.4744186, -68.1741714, 46.3096657, -114.6190186, 114.6485901
17: -110.2029495, 44.1483955, -109.9621124, 43.7173996, -153.9203491, 154.1105042
18: -58.8269386, 46.7583961, -58.7240601, 46.5986404, -105.4255829, 105.4824524
19: -46.8331070, 22.8346882, -46.7358551, 22.7312031, -69.5643082, 69.5705414
20: -40.6758728, 32.3435745, -40.5611649, 32.2893677, -72.9652405, 72.9047394
21: -61.4753838, 29.6907883, -61.3815765, 29.5578613, -91.0332413, 91.0723648
22: -67.8218536, 26.6775417, -67.7339478, 26.5603466, -94.3822021, 94.4114838
23: -45.7209892, 33.9759674, -45.6389542, 33.8544006, -79.5753860, 79.6149216
24: -60.3684349, 36.3537903, -60.2346153, 36.3169022, -96.6853333, 96.5884094
25: -47.1729202, 34.1464348, -47.0955467, 34.0685921, -81.2415085, 81.2419815
26: -66.6133041, 52.3645630, -66.5279541, 52.1826935, -118.7959900, 118.8925171
27: -58.9264107, 38.1474152, -58.7512398, 38.0911026, -97.0175171, 96.8986435
28: -44.0212555, 37.2988892, -43.9402466, 37.2507401, -81.2719879, 81.2391357
29: -76.0225449, 29.6651421, -75.9366913, 29.5145588, -105.5371017, 105.6018219
30: -56.4256554, 43.2500229, -56.3486595, 43.1750107, -99.6006622, 99.5986786
31: -61.6305504, 31.8196278, -61.5088501, 31.7322617, -93.3628082, 93.3284760
32: -48.2936211, 38.9082031, -48.2101517, 38.7661171, -87.0597382, 87.1183472
33: -71.8873901, 53.7565384, -71.7322922, 53.6952782, -125.5826645, 125.4888229
34: -59.1824417, 45.3338242, -59.0675545, 45.2667503, -104.4491882, 104.4013824
35: -66.6992264, 46.9584846, -66.5625763, 46.8985901, -113.5978088, 113.5210571
36: -58.9712334, 48.7085495, -58.8840103, 48.6435013, -107.6147308, 107.5925522
37: -76.3666382, 48.9608307, -76.2350082, 48.7815857, -125.1482239, 125.1958389
38: -71.0129852, 57.3608322, -70.8725433, 57.2363167, -128.2492981, 128.2333679
39: -88.3646851, 51.0477753, -88.2224274, 50.9897575, -139.3544312, 139.2702026
40: -59.1473732, 46.7780075, -58.9732170, 46.7260551, -105.8734207, 105.7512207
41: -43.8175774, 42.2348747, -43.6940536, 42.0782013, -85.8957825, 85.9289246
42: -32.9440155, 40.7970352, -32.8025742, 40.5702209, -73.5142365, 73.5996017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1624

## Relational analysis of IS_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2208253, upper bound: 59.3611036
time: 149.21 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3358983, upper bound: 59.3695868
time: 113.80 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -85.1830673, 35.1761932, -85.1871490, 35.1819839, -120.3650513, 120.3633423
1: -48.2246056, 33.1344604, -48.2267456, 33.1382103, -81.3628159, 81.3612061
2: -41.1408997, 28.6025009, -41.1471939, 28.6073380, -69.7482376, 69.7496948
3: -46.5412598, 38.4186935, -46.5439148, 38.4299393, -84.9711990, 84.9626083
4: -53.6280365, 36.7740898, -53.6399384, 36.7767563, -90.4047852, 90.4140320
5: -46.2641068, 40.4817085, -46.2663078, 40.4921722, -86.7562790, 86.7480087
6: -42.0587540, 40.9775848, -42.0617981, 40.9833984, -83.0421448, 83.0393829
7: -54.5928192, 41.0919991, -54.5951233, 41.0982132, -95.6910248, 95.6871185
8: -56.8120613, 48.1521454, -56.8260574, 48.1562500, -104.9683075, 104.9781952
9: -44.0672913, 41.3480873, -44.0692978, 41.3527222, -85.4200134, 85.4173813
10: -65.4886017, 52.9772530, -65.4924927, 52.9863739, -118.4749603, 118.4697418
11: -62.5968552, 40.8829880, -62.5998878, 40.8887749, -103.4856262, 103.4828796
12: -50.6276131, 49.9165878, -50.6312904, 49.9282494, -100.5558624, 100.5478821
13: -64.4948044, 55.0509262, -64.5007172, 55.0599594, -119.5547638, 119.5516357
14: -104.3403778, 24.1553955, -104.3471298, 24.1647511, -128.5050964, 128.5025330
15: -54.4290962, 33.1288109, -54.4340134, 33.1313591, -87.5604553, 87.5628204
16: -68.3093491, 46.4744186, -68.3131866, 46.4820709, -114.7914200, 114.7876053
17: -110.2029495, 44.1483955, -110.2080917, 44.1612968, -154.3642426, 154.3564758
18: -58.8269386, 46.7583961, -58.8357010, 46.7617874, -105.5887222, 105.5940857
19: -46.8331070, 22.8346882, -46.8401031, 22.8376961, -69.6708069, 69.6747894
20: -40.6758728, 32.3435745, -40.6815414, 32.3461914, -73.0220642, 73.0251160
21: -61.4753838, 29.6907883, -61.4828262, 29.6943436, -91.1697235, 91.1736145
22: -67.8218536, 26.6775417, -67.8306351, 26.6793900, -94.5012360, 94.5081711
23: -45.7209892, 33.9759674, -45.7236252, 33.9793663, -79.7003555, 79.6995926
24: -60.3684349, 36.3537903, -60.3767242, 36.3554611, -96.7238846, 96.7305145
25: -47.1729202, 34.1464348, -47.1786270, 34.1482391, -81.3211594, 81.3250580
26: -66.6133041, 52.3645630, -66.6275482, 52.3688087, -118.9821167, 118.9920883
27: -58.9264107, 38.1474152, -58.9359779, 38.1505699, -97.0769730, 97.0833893
28: -44.0212555, 37.2988892, -44.0254593, 37.3012543, -81.3225098, 81.3243484
29: -76.0225449, 29.6651421, -76.0287170, 29.6679440, -105.6904907, 105.6938553
30: -56.4256554, 43.2500229, -56.4315109, 43.2546158, -99.6802673, 99.6815338
31: -61.6305504, 31.8196278, -61.6389771, 31.8241310, -93.4546738, 93.4586029
32: -48.2936211, 38.9082031, -48.2973480, 38.9131012, -87.2067108, 87.2055435
33: -71.8873901, 53.7565384, -71.8938904, 53.7634430, -125.6508179, 125.6504288
34: -59.1824417, 45.3338242, -59.1896210, 45.3449249, -104.5273666, 104.5234375
35: -66.6992264, 46.9584846, -66.7051697, 46.9622116, -113.6614380, 113.6636505
36: -58.9712334, 48.7085495, -58.9762344, 48.7103004, -107.6815338, 107.6847763
37: -76.3666382, 48.9608307, -76.3762360, 48.9624977, -125.3291321, 125.3370667
38: -71.0129852, 57.3608322, -71.0202637, 57.3697472, -128.3827362, 128.3810883
39: -88.3646851, 51.0477753, -88.3753510, 51.0496864, -139.4143677, 139.4231262
40: -59.1473732, 46.7780075, -59.1556587, 46.7810211, -105.9283905, 105.9336700
41: -43.8175774, 42.2348747, -43.8213196, 42.2392960, -86.0568695, 86.0561905
42: -32.9440155, 40.7970352, -32.9456482, 40.8047333, -73.7487488, 73.7426834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1624

## Relational analysis of IS_B2_A2_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2848914, upper bound: 59.3611039
time: 593.92 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2237801, upper bound: 59.3695870
time: 111.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 707.58 seconds
IS_B1_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 707.58
Output dim: 6, lower bound: -59.2208253, upper bound: 59.3611036
IS_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 707.58
Output dim: 6, lower bound: -59.3358983, upper bound: 59.3695868
IS_B2_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 707.58
Output dim: 6, lower bound: -59.2848914, upper bound: 59.3611039
IS_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 707.58
Output dim: 6, lower bound: -59.2237801, upper bound: 59.3695870

## BFS IS instance: IS_B1_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -85.1829529, 35.1802139, -85.0386505, 35.0978851, -120.2808380, 120.2188568
1: -48.2221031, 33.1392899, -48.1194000, 33.0668449, -81.2889404, 81.2586823
2: -41.1375885, 28.6155262, -40.9435196, 28.5388336, -69.6764145, 69.5590439
3: -46.5322647, 38.4459190, -46.3237000, 38.3408585, -84.8731232, 84.7696228
4: -53.6278877, 36.7829933, -53.3968811, 36.6733017, -90.3011856, 90.1798706
5: -46.2647209, 40.5254593, -46.0660744, 40.4038696, -86.6685944, 86.5915298
6: -42.0623169, 40.9951248, -41.9430313, 40.8018951, -82.8642120, 82.9381485
7: -54.5833054, 41.0988159, -54.4456673, 41.0476036, -95.6309052, 95.5444794
8: -56.8119507, 48.1619453, -56.5746231, 48.0285416, -104.8404922, 104.7365723
9: -44.0632401, 41.4054794, -44.0059929, 41.1912918, -85.2545319, 85.4114685
10: -65.4848099, 53.0049973, -65.3427429, 52.6383095, -118.1231079, 118.3477402
11: -62.6440048, 40.8738556, -62.4763603, 40.6469803, -103.2909775, 103.3502197
12: -50.6309853, 49.9132767, -50.4446564, 49.4990883, -100.1300659, 100.3579330
13: -64.4818802, 55.1220284, -64.4205246, 54.8801079, -119.3619843, 119.5425568
14: -104.3933411, 24.1433086, -104.1948318, 23.8969402, -128.2902679, 128.3381348
15: -54.4329872, 33.1239929, -54.3157196, 33.0229340, -87.4559174, 87.4397125
16: -68.3120422, 46.4900360, -68.1665955, 46.3080215, -114.6200638, 114.6566238
17: -110.2960510, 44.1361542, -109.9569397, 43.7073936, -154.0034485, 154.0930786
18: -58.8516083, 46.7106857, -58.7201233, 46.5772171, -105.4288254, 105.4308090
19: -46.8849716, 22.8299294, -46.7340698, 22.7269402, -69.6119080, 69.5639954
20: -40.6885071, 32.3371010, -40.5591164, 32.2850113, -72.9735184, 72.8962173
21: -61.5307961, 29.6800117, -61.3795013, 29.5520458, -91.0828400, 91.0595093
22: -67.9176331, 26.6651211, -67.7317810, 26.5527039, -94.4703369, 94.3969040
23: -45.7641220, 33.9686508, -45.6373444, 33.8496323, -79.6137543, 79.6059952
24: -60.4562721, 36.3448181, -60.2326393, 36.3096428, -96.7659073, 96.5774536
25: -47.2246170, 34.1370697, -47.0936356, 34.0628052, -81.2874222, 81.2307053
26: -66.6692047, 52.3528404, -66.5248337, 52.1749039, -118.8441086, 118.8776703
27: -59.0197334, 38.1325226, -58.7487755, 38.0834122, -97.1031342, 96.8812866
28: -44.0785141, 37.2907715, -43.9385986, 37.2453918, -81.3239059, 81.2293625
29: -76.1112823, 29.6496124, -75.9346619, 29.5059223, -105.6171875, 105.5842743
30: -56.4864807, 43.2375107, -56.3469887, 43.1674805, -99.6539536, 99.5844955
31: -61.6953964, 31.8125839, -61.5060120, 31.7266121, -93.4219971, 93.3185959
32: -48.2990532, 38.9165115, -48.2049103, 38.7639160, -87.0629730, 87.1214218
33: -71.8924103, 53.7626266, -71.7264862, 53.6930008, -125.5854111, 125.4891129
34: -59.1932831, 45.3353577, -59.0640450, 45.2635880, -104.4568634, 104.3993988
35: -66.7094574, 46.9599380, -66.5589600, 46.8960342, -113.6054688, 113.5188980
36: -58.9813385, 48.6949387, -58.8807983, 48.6359329, -107.6172714, 107.5757370
37: -76.3784637, 48.9524345, -76.2309570, 48.7767410, -125.1552048, 125.1833954
38: -71.0178223, 57.3583603, -70.8662796, 57.2297707, -128.2475739, 128.2246399
39: -88.3652649, 51.0672760, -88.2164307, 50.9885559, -139.3538208, 139.2837067
40: -59.1521568, 46.8003693, -58.9676933, 46.7251740, -105.8773193, 105.7680664
41: -43.8251038, 42.2423248, -43.6899414, 42.0764160, -85.9015198, 85.9322662
42: -32.9448814, 40.8174820, -32.7974548, 40.5676880, -73.5125732, 73.6149292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3155911, upper bound: 59.2638544
time: 112.46 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2007949, upper bound: 59.3664737
time: 201.93 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -85.1829529, 35.1802139, -85.1801300, 35.1789703, -120.3619232, 120.3603363
1: -48.2221031, 33.1392899, -48.2223816, 33.1366348, -81.3587341, 81.3616714
2: -41.1375885, 28.6155262, -41.1442947, 28.6057796, -69.7433624, 69.7598190
3: -46.5322647, 38.4459190, -46.5363693, 38.4277191, -84.9599762, 84.9822845
4: -53.6278877, 36.7829933, -53.6364899, 36.7735672, -90.4014587, 90.4194794
5: -46.2647209, 40.5254593, -46.2632370, 40.4901199, -86.7548370, 86.7886887
6: -42.0623169, 40.9951248, -42.0575294, 40.9817543, -83.0440674, 83.0526581
7: -54.5833054, 41.0988159, -54.5884399, 41.0964813, -95.6797867, 95.6872559
8: -56.8119507, 48.1619453, -56.8227501, 48.1541901, -104.9661407, 104.9846954
9: -44.0632401, 41.4054794, -44.0634308, 41.3505173, -85.4137573, 85.4689102
10: -65.4848099, 53.0049973, -65.4869080, 52.9835243, -118.4683380, 118.4918976
11: -62.6440048, 40.8738556, -62.5977173, 40.8827553, -103.5267563, 103.4715576
12: -50.6309853, 49.9132767, -50.6267433, 49.9232407, -100.5542221, 100.5400162
13: -64.4818802, 55.1220284, -64.4926147, 55.0573273, -119.5392075, 119.6146393
14: -104.3933411, 24.1433086, -104.3418274, 24.1570549, -128.5503998, 128.4851379
15: -54.4329872, 33.1239929, -54.4313354, 33.1249313, -87.5579224, 87.5553284
16: -68.3120422, 46.4900360, -68.3056183, 46.4804115, -114.7924500, 114.7956543
17: -110.2960510, 44.1361542, -110.2029572, 44.1513252, -154.4473724, 154.3391113
18: -58.8516083, 46.7106857, -58.8317566, 46.7403526, -105.5919647, 105.5424423
19: -46.8849716, 22.8299294, -46.8383179, 22.8334351, -69.7183990, 69.6682434
20: -40.6885071, 32.3371010, -40.6794815, 32.3418312, -73.0303345, 73.0165863
21: -61.5307961, 29.6800117, -61.4807587, 29.6885338, -91.2193298, 91.1607666
22: -67.9176331, 26.6651211, -67.8284760, 26.6717434, -94.5893631, 94.4935913
23: -45.7641220, 33.9686508, -45.7220078, 33.9745865, -79.7387085, 79.6906586
24: -60.4562721, 36.3448181, -60.3747559, 36.3482170, -96.8044891, 96.7195740
25: -47.2246170, 34.1370697, -47.1767159, 34.1424637, -81.3670731, 81.3137817
26: -66.6692047, 52.3528404, -66.6244278, 52.3610420, -119.0302353, 118.9772644
27: -59.0197334, 38.1325226, -58.9335136, 38.1428452, -97.1625671, 97.0660324
28: -44.0785141, 37.2907715, -44.0238113, 37.2959137, -81.3744278, 81.3145828
29: -76.1112823, 29.6496124, -76.0267029, 29.6593227, -105.7705994, 105.6763153
30: -56.4864807, 43.2375107, -56.4298439, 43.2470818, -99.7335510, 99.6673584
31: -61.6953964, 31.8125839, -61.6361389, 31.8185005, -93.5139008, 93.4487228
32: -48.2990532, 38.9165115, -48.2920952, 38.9109039, -87.2099533, 87.2086029
33: -71.8924103, 53.7626266, -71.8880768, 53.7611542, -125.6535492, 125.6507034
34: -59.1932831, 45.3353577, -59.1860962, 45.3417587, -104.5350418, 104.5214462
35: -66.7094574, 46.9599380, -66.7015457, 46.9596634, -113.6691055, 113.6614838
36: -58.9813385, 48.6949387, -58.9730110, 48.7027092, -107.6840515, 107.6679459
37: -76.3784637, 48.9524345, -76.3722153, 48.9577217, -125.3361816, 125.3246460
38: -71.0178223, 57.3583603, -71.0139465, 57.3631897, -128.3810120, 128.3722992
39: -88.3652649, 51.0672760, -88.3693695, 51.0484581, -139.4137268, 139.4366455
40: -59.1521568, 46.8003693, -59.1501656, 46.7801437, -105.9322968, 105.9505310
41: -43.8251038, 42.2423248, -43.8172073, 42.2375031, -86.0626068, 86.0595322
42: -32.9448814, 40.8174820, -32.9405136, 40.8021965, -73.7470779, 73.7579880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3161534, upper bound: 59.2638545
time: 121.36 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3332442, upper bound: 59.3664740
time: 126.15 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 249.90 seconds
IS_B1_A2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 249.90
Output dim: 6, lower bound: -59.3155911, upper bound: 59.2638544
IS_B1_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 249.90
Output dim: 6, lower bound: -59.2007949, upper bound: 59.3664737
IS_B2_A2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 249.90
Output dim: 6, lower bound: -59.3161534, upper bound: 59.2638545
IS_B2_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 249.90
Output dim: 6, lower bound: -59.3332442, upper bound: 59.3664740

## BFS IS instance: IS_B1_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -85.1828461, 35.1801682, -85.0331573, 35.0947495, -120.2775955, 120.2133179
1: -48.2220535, 33.1391907, -48.1173477, 33.0620346, -81.2840881, 81.2565384
2: -41.1375542, 28.6154251, -40.9416046, 28.5337696, -69.6713257, 69.5570297
3: -46.5322342, 38.4457550, -46.3217430, 38.3330040, -84.8652267, 84.7674942
4: -53.6278572, 36.7828560, -53.3949509, 36.6663246, -90.2941666, 90.1778030
5: -46.2646866, 40.5253601, -46.0637512, 40.3984146, -86.6630936, 86.5891113
6: -42.0621567, 40.9950790, -41.9349327, 40.7996483, -82.8618011, 82.9300079
7: -54.5832481, 41.0986900, -54.4432220, 41.0413399, -95.6245804, 95.5419159
8: -56.8119202, 48.1617508, -56.5730438, 48.0179405, -104.8298645, 104.7347946
9: -44.0631943, 41.4054413, -44.0037766, 41.1890450, -85.2522354, 85.4092102
10: -65.4846725, 53.0049515, -65.3353500, 52.6358299, -118.1204758, 118.3403015
11: -62.6438370, 40.8738136, -62.4674644, 40.6447067, -103.2885437, 103.3412781
12: -50.6308441, 49.9132347, -50.4370880, 49.4963913, -100.1272278, 100.3503189
13: -64.4817963, 55.1219559, -64.4165344, 54.8765144, -119.3582993, 119.5384827
14: -104.3931732, 24.1432400, -104.1859589, 23.8937206, -128.2868805, 128.3291931
15: -54.4329262, 33.1237411, -54.3124352, 33.0101433, -87.4430695, 87.4361725
16: -68.3118210, 46.4899750, -68.1568527, 46.3054886, -114.6173096, 114.6468201
17: -110.2959671, 44.1360893, -109.9521255, 43.7036934, -153.9996490, 154.0882111
18: -58.8514557, 46.7106400, -58.7123566, 46.5751266, -105.4265823, 105.4229889
19: -46.8848953, 22.8299026, -46.7303047, 22.7255192, -69.6104126, 69.5602036
20: -40.6884422, 32.3370667, -40.5561066, 32.2832947, -72.9717331, 72.8931732
21: -61.5306931, 29.6799889, -61.3738670, 29.5503235, -91.0810165, 91.0538483
22: -67.9175491, 26.6648674, -67.7272644, 26.5400352, -94.4575806, 94.3921356
23: -45.7639618, 33.9686127, -45.6293030, 33.8478775, -79.6118393, 79.5979080
24: -60.4561462, 36.3447914, -60.2264442, 36.3074646, -96.7636108, 96.5712357
25: -47.2245178, 34.1370316, -47.0885239, 34.0607719, -81.2852859, 81.2255554
26: -66.6690979, 52.3527870, -66.5197678, 52.1718369, -118.8409348, 118.8725433
27: -59.0196762, 38.1324654, -58.7458458, 38.0811844, -97.1008453, 96.8783112
28: -44.0784302, 37.2907257, -43.9343376, 37.2431602, -81.3215866, 81.2250671
29: -76.1111832, 29.6494484, -75.9298401, 29.4976463, -105.6088257, 105.5792847
30: -56.4864159, 43.2374535, -56.3434486, 43.1646461, -99.6510544, 99.5809021
31: -61.6951866, 31.8125267, -61.4978256, 31.7241077, -93.4192963, 93.3103485
32: -48.2989578, 38.9164696, -48.2002220, 38.7622299, -87.0611877, 87.1166916
33: -71.8923264, 53.7625847, -71.7218933, 53.6909790, -125.5833054, 125.4844818
34: -59.1931915, 45.3353043, -59.0596962, 45.2611046, -104.4542999, 104.3950043
35: -66.7093811, 46.9599152, -66.5552673, 46.8945045, -113.6038818, 113.5151825
36: -58.9812622, 48.6949158, -58.8775673, 48.6347008, -107.6159668, 107.5724792
37: -76.3782654, 48.9524155, -76.2213135, 48.7752151, -125.1534805, 125.1737213
38: -71.0177383, 57.3583183, -70.8620224, 57.2275009, -128.2452393, 128.2203369
39: -88.3651733, 51.0672302, -88.2121201, 50.9864540, -139.3516083, 139.2793579
40: -59.1519966, 46.8003273, -58.9603767, 46.7231636, -105.8751526, 105.7607040
41: -43.8249969, 42.2422905, -43.6839485, 42.0747910, -85.8997803, 85.9262390
42: -32.9447670, 40.8174362, -32.7916985, 40.5654755, -73.5102386, 73.6091309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.1160299, upper bound: 59.3490608
time: 99.03 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2305299, upper bound: 59.3664741
time: 134.27 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -85.1828461, 35.1801682, -85.1746521, 35.1758270, -120.3586731, 120.3548126
1: -48.2220535, 33.1391907, -48.2203293, 33.1318130, -81.3538666, 81.3595200
2: -41.1375542, 28.6154251, -41.1423759, 28.6007309, -69.7382812, 69.7577972
3: -46.5322342, 38.4457550, -46.5344086, 38.4198723, -84.9521027, 84.9801636
4: -53.6278572, 36.7828560, -53.6345673, 36.7665863, -90.3944321, 90.4174194
5: -46.2646866, 40.5253601, -46.2609177, 40.4846573, -86.7493439, 86.7862778
6: -42.0621567, 40.9950790, -42.0494423, 40.9795074, -83.0416641, 83.0445175
7: -54.5832481, 41.0986900, -54.5860023, 41.0902214, -95.6734619, 95.6846924
8: -56.8119202, 48.1617508, -56.8211746, 48.1436081, -104.9555283, 104.9829254
9: -44.0631943, 41.4054413, -44.0612183, 41.3482819, -85.4114685, 85.4666519
10: -65.4846725, 53.0049515, -65.4794922, 52.9810257, -118.4656830, 118.4844437
11: -62.6438370, 40.8738136, -62.5888214, 40.8805046, -103.5243378, 103.4626312
12: -50.6308441, 49.9132347, -50.6191826, 49.9205322, -100.5513763, 100.5324173
13: -64.4817963, 55.1219559, -64.4886475, 55.0537300, -119.5355072, 119.6106033
14: -104.3931732, 24.1432400, -104.3329849, 24.1538086, -128.5469818, 128.4762268
15: -54.4329262, 33.1237411, -54.4280319, 33.1121674, -87.5450897, 87.5517731
16: -68.3118210, 46.4899750, -68.2958221, 46.4778786, -114.7896957, 114.7857895
17: -110.2959671, 44.1360893, -110.1981201, 44.1476021, -154.4435425, 154.3342133
18: -58.8514557, 46.7106400, -58.8239975, 46.7382584, -105.5897141, 105.5346375
19: -46.8848953, 22.8299026, -46.8345451, 22.8320160, -69.7169037, 69.6644440
20: -40.6884422, 32.3370667, -40.6764755, 32.3401070, -73.0285492, 73.0135422
21: -61.5306931, 29.6799889, -61.4751167, 29.6868019, -91.2174988, 91.1551056
22: -67.9175491, 26.6648674, -67.8239594, 26.6590519, -94.5765991, 94.4888306
23: -45.7639618, 33.9686127, -45.7139702, 33.9728432, -79.7368011, 79.6825867
24: -60.4561462, 36.3447914, -60.3685532, 36.3460388, -96.8021851, 96.7133408
25: -47.2245178, 34.1370316, -47.1716156, 34.1404190, -81.3649368, 81.3086472
26: -66.6690979, 52.3527870, -66.6193466, 52.3579941, -119.0270920, 118.9721298
27: -59.0196762, 38.1324654, -58.9305840, 38.1406288, -97.1603012, 97.0630493
28: -44.0784302, 37.2907257, -44.0195541, 37.2936821, -81.3721085, 81.3102798
29: -76.1111832, 29.6494484, -76.0218506, 29.6510239, -105.7622070, 105.6712952
30: -56.4864159, 43.2374535, -56.4263039, 43.2442474, -99.7306671, 99.6637573
31: -61.6951866, 31.8125267, -61.6279144, 31.8159637, -93.5111389, 93.4404449
32: -48.2989578, 38.9164696, -48.2874184, 38.9092064, -87.2081604, 87.2038879
33: -71.8923264, 53.7625847, -71.8834839, 53.7591438, -125.6514587, 125.6460724
34: -59.1931915, 45.3353043, -59.1817398, 45.3392830, -104.5324707, 104.5170441
35: -66.7093811, 46.9599152, -66.6978607, 46.9581261, -113.6675110, 113.6577759
36: -58.9812622, 48.6949158, -58.9697838, 48.7014694, -107.6827240, 107.6646957
37: -76.3782654, 48.9524155, -76.3625641, 48.9561462, -125.3344116, 125.3149643
38: -71.0177383, 57.3583183, -71.0097046, 57.3609314, -128.3786621, 128.3680267
39: -88.3651733, 51.0672302, -88.3650360, 51.0463562, -139.4115295, 139.4322510
40: -59.1519966, 46.8003273, -59.1428299, 46.7781219, -105.9301147, 105.9431610
41: -43.8249969, 42.2422905, -43.8112183, 42.2358932, -86.0608826, 86.0535126
42: -32.9447670, 40.8174362, -32.9347687, 40.7999878, -73.7447510, 73.7521973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1031
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1087
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1071
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2310910, upper bound: 59.3490612
time: 491.06 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2310910, upper bound: 59.3664742
time: 85.35 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 578.82 seconds
IS_B1_A2_B2_A2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 578.82
Output dim: 6, lower bound: -59.1160299, upper bound: 59.3490608
IS_B1_A2_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 578.82
Output dim: 6, lower bound: -59.2305299, upper bound: 59.3664741
IS_B2_A2_B2_A2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 578.82
Output dim: 6, lower bound: -59.2310910, upper bound: 59.3490612
IS_B2_A2_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 578.82
Output dim: 6, lower bound: -59.2310910, upper bound: 59.3664742

## BFS IS instance: IS_B1_A2_B2_A2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.1774597, 35.1771011, -85.0331573, 35.0947495, -120.2722015, 120.2102509
1: -48.2200546, 33.1344757, -48.1173477, 33.0620346, -81.2820892, 81.2518234
2: -41.1356697, 28.6104622, -40.9416046, 28.5337696, -69.6694260, 69.5520630
3: -46.5303078, 38.4380608, -46.3217430, 38.3330040, -84.8633118, 84.7597961
4: -53.6259689, 36.7760239, -53.3949509, 36.6663246, -90.2922897, 90.1709747
5: -46.2623978, 40.5199966, -46.0637512, 40.3984146, -86.6608124, 86.5837479
6: -42.0542221, 40.9928818, -41.9349327, 40.7996483, -82.8538666, 82.9278107
7: -54.5808830, 41.0925446, -54.4432220, 41.0413399, -95.6222153, 95.5357590
8: -56.8103867, 48.1513596, -56.5730438, 48.0179405, -104.8283234, 104.7244034
9: -44.0610237, 41.4032669, -44.0037766, 41.1890450, -85.2500687, 85.4070282
10: -65.4774017, 53.0025253, -65.3353500, 52.6358299, -118.1132355, 118.3378677
11: -62.6351242, 40.8715858, -62.4674644, 40.6447067, -103.2798309, 103.3390503
12: -50.6234322, 49.9105606, -50.4370880, 49.4963913, -100.1198120, 100.3476486
13: -64.4779205, 55.1184311, -64.4165344, 54.8765144, -119.3544235, 119.5349579
14: -104.3844986, 24.1400757, -104.1859589, 23.8937206, -128.2782135, 128.3260193
15: -54.4296951, 33.1112137, -54.3124352, 33.0101433, -87.4398346, 87.4236450
16: -68.3022614, 46.4874992, -68.1568527, 46.3054886, -114.6077499, 114.6443481
17: -110.2912521, 44.1324081, -109.9521255, 43.7036934, -153.9949493, 154.0845337
18: -58.8438530, 46.7085838, -58.7123566, 46.5751266, -105.4189758, 105.4209366
19: -46.8811874, 22.8285027, -46.7303047, 22.7255192, -69.6067047, 69.5588074
20: -40.6854935, 32.3353806, -40.5561066, 32.2832947, -72.9687881, 72.8914871
21: -61.5251617, 29.6782722, -61.3738670, 29.5503235, -91.0754776, 91.0521393
22: -67.9131317, 26.6524525, -67.7272644, 26.5400352, -94.4531708, 94.3797150
23: -45.7561111, 33.9669151, -45.6293030, 33.8478775, -79.6039886, 79.5962143
24: -60.4500618, 36.3426514, -60.2264442, 36.3074646, -96.7575226, 96.5690918
25: -47.2195129, 34.1350403, -47.0885239, 34.0607719, -81.2802887, 81.2235641
26: -66.6641235, 52.3498077, -66.5197678, 52.1718369, -118.8359604, 118.8695679
27: -59.0168190, 38.1302910, -58.7458458, 38.0811844, -97.0980072, 96.8761292
28: -44.0742645, 37.2885399, -43.9343376, 37.2431602, -81.3174210, 81.2228775
29: -76.1064453, 29.6413383, -75.9298401, 29.4976463, -105.6040955, 105.5711823
30: -56.4829369, 43.2346725, -56.3434486, 43.1646461, -99.6475754, 99.5781250
31: -61.6871796, 31.8100452, -61.4978256, 31.7241077, -93.4112854, 93.3078690
32: -48.2943687, 38.9148102, -48.2002220, 38.7622299, -87.0565948, 87.1150284
33: -71.8878174, 53.7606201, -71.7218933, 53.6909790, -125.5787964, 125.4825134
34: -59.1889114, 45.3328781, -59.0596962, 45.2611046, -104.4500122, 104.3925781
35: -66.7057648, 46.9584198, -66.5552673, 46.8945045, -113.6002655, 113.5136871
36: -58.9780884, 48.6937027, -58.8775673, 48.6347008, -107.6127930, 107.5712738
37: -76.3688126, 48.9508896, -76.2213135, 48.7752151, -125.1440201, 125.1721954
38: -71.0135345, 57.3561211, -70.8620224, 57.2275009, -128.2410278, 128.2181396
39: -88.3609314, 51.0651703, -88.2121201, 50.9864540, -139.3473816, 139.2772827
40: -59.1448059, 46.7983551, -58.9603767, 46.7231636, -105.8679657, 105.7587280
41: -43.8191147, 42.2407074, -43.6839485, 42.0747910, -85.8939056, 85.9246521
42: -32.9391327, 40.8152618, -32.7916985, 40.5654755, -73.5046082, 73.6069565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=194, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1054
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1055
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1135
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1171
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1151
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1038
type: A, layer: 1, pos: 1038

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.1835098, upper bound: 59.2934064
time: 121.65 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.0689102, upper bound: 59.2172141
time: 103.27 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.1774597, 35.1771011, -85.1746521, 35.1758270, -120.3532867, 120.3517532
1: -48.2200546, 33.1344757, -48.2203293, 33.1318130, -81.3518677, 81.3548050
2: -41.1356697, 28.6104622, -41.1423759, 28.6007309, -69.7363968, 69.7528381
3: -46.5303078, 38.4380608, -46.5344086, 38.4198723, -84.9501801, 84.9724579
4: -53.6259689, 36.7760239, -53.6345673, 36.7665863, -90.3925552, 90.4105835
5: -46.2623978, 40.5199966, -46.2609177, 40.4846573, -86.7470551, 86.7809143
6: -42.0542221, 40.9928818, -42.0494423, 40.9795074, -83.0337296, 83.0423203
7: -54.5808830, 41.0925446, -54.5860023, 41.0902214, -95.6710968, 95.6785431
8: -56.8103867, 48.1513596, -56.8211746, 48.1436081, -104.9539948, 104.9725342
9: -44.0610237, 41.4032669, -44.0612183, 41.3482819, -85.4093018, 85.4644852
10: -65.4774017, 53.0025253, -65.4794922, 52.9810257, -118.4584274, 118.4820175
11: -62.6351242, 40.8715858, -62.5888214, 40.8805046, -103.5156250, 103.4604034
12: -50.6234322, 49.9105606, -50.6191826, 49.9205322, -100.5439606, 100.5297394
13: -64.4779205, 55.1184311, -64.4886475, 55.0537300, -119.5316467, 119.6070786
14: -104.3844986, 24.1400757, -104.3329849, 24.1538086, -128.5382996, 128.4730530
15: -54.4296951, 33.1112137, -54.4280319, 33.1121674, -87.5418472, 87.5392456
16: -68.3022614, 46.4874992, -68.2958221, 46.4778786, -114.7801361, 114.7833252
17: -110.2912521, 44.1324081, -110.1981201, 44.1476021, -154.4388580, 154.3305359
18: -58.8438530, 46.7085838, -58.8239975, 46.7382584, -105.5821075, 105.5325775
19: -46.8811874, 22.8285027, -46.8345451, 22.8320160, -69.7132034, 69.6630478
20: -40.6854935, 32.3353806, -40.6764755, 32.3401070, -73.0256042, 73.0118561
21: -61.5251617, 29.6782722, -61.4751167, 29.6868019, -91.2119598, 91.1533890
22: -67.9131317, 26.6524525, -67.8239594, 26.6590519, -94.5721817, 94.4764099
23: -45.7561111, 33.9669151, -45.7139702, 33.9728432, -79.7289581, 79.6808777
24: -60.4500618, 36.3426514, -60.3685532, 36.3460388, -96.7960968, 96.7112045
25: -47.2195129, 34.1350403, -47.1716156, 34.1404190, -81.3599319, 81.3066483
26: -66.6641235, 52.3498077, -66.6193466, 52.3579941, -119.0221176, 118.9691544
27: -59.0168190, 38.1302910, -58.9305840, 38.1406288, -97.1574478, 97.0608749
28: -44.0742645, 37.2885399, -44.0195541, 37.2936821, -81.3679428, 81.3080902
29: -76.1064453, 29.6413383, -76.0218506, 29.6510239, -105.7574692, 105.6631775
30: -56.4829369, 43.2346725, -56.4263039, 43.2442474, -99.7271881, 99.6609802
31: -61.6871796, 31.8100452, -61.6279144, 31.8159637, -93.5031433, 93.4379578
32: -48.2943687, 38.9148102, -48.2874184, 38.9092064, -87.2035751, 87.2022247
33: -71.8878174, 53.7606201, -71.8834839, 53.7591438, -125.6469421, 125.6441040
34: -59.1889114, 45.3328781, -59.1817398, 45.3392830, -104.5281982, 104.5146179
35: -66.7057648, 46.9584198, -66.6978607, 46.9581261, -113.6638947, 113.6562805
36: -58.9780884, 48.6937027, -58.9697838, 48.7014694, -107.6795502, 107.6634827
37: -76.3688126, 48.9508896, -76.3625641, 48.9561462, -125.3249588, 125.3134460
38: -71.0135345, 57.3561211, -71.0097046, 57.3609314, -128.3744507, 128.3658142
39: -88.3609314, 51.0651703, -88.3650360, 51.0463562, -139.4072876, 139.4301910
40: -59.1448059, 46.7983551, -59.1428299, 46.7781219, -105.9229279, 105.9411850
41: -43.8191147, 42.2407074, -43.8112183, 42.2358932, -86.0549927, 86.0519257
42: -32.9391327, 40.8152618, -32.9347687, 40.7999878, -73.7391205, 73.7500229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1031
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1087
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1102
type: B, layer: 1, pos: 1102
type: A, layer: 1, pos: 1032
type: B, layer: 1, pos: 1032
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1267
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1029
type: B, layer: 1, pos: 1029
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1033
type: B, layer: 1, pos: 1033
type: A, layer: 1, pos: 1246
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1262
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1118
type: B, layer: 1, pos: 1118
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1258
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1030
type: B, layer: 1, pos: 1030
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1103
type: B, layer: 1, pos: 1103
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1076
type: A, layer: 1, pos: 1076
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1119
type: B, layer: 1, pos: 1119
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1071
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1245
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1214
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1027
type: B, layer: 1, pos: 1027
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1034
type: B, layer: 1, pos: 1034
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1055
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1028
type: B, layer: 1, pos: 1028
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1035
type: B, layer: 1, pos: 1035
type: A, layer: 1, pos: 1135
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1260
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1171
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1236
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1151
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1183
type: A, layer: 1, pos: 1183
type: B, layer: 1, pos: 1199
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1275
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1024
type: B, layer: 1, pos: 1024
type: A, layer: 1, pos: 1167
type: B, layer: 1, pos: 1167
type: A, layer: 1, pos: 1037
type: B, layer: 1, pos: 1037
type: A, layer: 1, pos: 1026
type: B, layer: 1, pos: 1026
type: A, layer: 1, pos: 1036
type: B, layer: 1, pos: 1036
type: A, layer: 1, pos: 1039
type: B, layer: 1, pos: 1039
type: A, layer: 1, pos: 1038
type: B, layer: 1, pos: 1038

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1703

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2081430, upper bound: 59.2616677
time: 196.32 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2081429, upper bound: 59.2963192
time: 101.23 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 299.91 seconds
IS_B1_A2_B2_A2_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 299.91
Output dim: 6, lower bound: -59.1835098, upper bound: 59.2934064
IS_B1_A2_B2_A2_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 299.91
Output dim: 6, lower bound: -59.0689102, upper bound: 59.2172141
IS_B2_A2_B2_A2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 299.91
Output dim: 6, lower bound: -59.2081430, upper bound: 59.2616677
IS_B2_A2_B2_A2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 9, time: 299.91
Output dim: 6, lower bound: -59.2081429, upper bound: 59.2963192

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 124.51 + 6428.80 = 6553.31 seconds
