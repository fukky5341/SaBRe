## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.89 + 125.35 = 128.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -59.3940217, upper bound: 59.3940217

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3853836, upper bound: 59.3853286
time: 103.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3853286, upper bound: 59.3853836
time: 195.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 299.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 299.55
Output dim: 6, lower bound: -59.3853836, upper bound: 59.3853286
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 299.55
Output dim: 6, lower bound: -59.3853286, upper bound: 59.3853836

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3755403, upper bound: 59.3710440
time: 107.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3711049, upper bound: 59.3754840
time: 139.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3754840, upper bound: 59.3711049
time: 207.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3710440, upper bound: 59.3755403
time: 136.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 345.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 345.80
Output dim: 6, lower bound: -59.3755403, upper bound: 59.3710440
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 345.80
Output dim: 6, lower bound: -59.3711049, upper bound: 59.3754840
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 345.80
Output dim: 6, lower bound: -59.3754840, upper bound: 59.3711049
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 345.80
Output dim: 6, lower bound: -59.3710440, upper bound: 59.3755403

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3706348, upper bound: 59.3587440
time: 110.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3628412, upper bound: 59.3661204
time: 133.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3661821, upper bound: 59.3632058
time: 126.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3583848, upper bound: 59.3705740
time: 122.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3705740, upper bound: 59.3583848
time: 151.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3632058, upper bound: 59.3661821
time: 115.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3661204, upper bound: 59.3628413
time: 174.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3587440, upper bound: 59.3706348
time: 118.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 295.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3706348, upper bound: 59.3587440
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3628412, upper bound: 59.3661204
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3661821, upper bound: 59.3632058
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3583848, upper bound: 59.3705740
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3705740, upper bound: 59.3583848
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3632058, upper bound: 59.3661821
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3661204, upper bound: 59.3628413
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.33
Output dim: 6, lower bound: -59.3587440, upper bound: 59.3706348

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3585259, upper bound: 59.3343187
time: 124.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3435034, upper bound: 59.3458017
time: 159.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3499472, upper bound: 59.3388690
time: 105.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3387342, upper bound: 59.3539975
time: 107.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3540690, upper bound: 59.3389283
time: 913.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3389881, upper bound: 59.3503042
time: 119.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3454647, upper bound: 59.3434162
time: 122.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3341296, upper bound: 59.3584607
time: 89.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3584607, upper bound: 59.3341296
time: 146.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3434162, upper bound: 59.3454647
time: 98.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3503042, upper bound: 59.3389881
time: 122.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3389283, upper bound: 59.3540690
time: 112.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3539974, upper bound: 59.3387342
time: 118.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3388690, upper bound: 59.3499472
time: 244.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3458017, upper bound: 59.3435034
time: 120.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3343187, upper bound: 59.3585259
time: 117.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 241.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3585259, upper bound: 59.3343187
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3435034, upper bound: 59.3458017
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3499472, upper bound: 59.3388690
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3387342, upper bound: 59.3539975
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3540690, upper bound: 59.3389283
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3389881, upper bound: 59.3503042
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3454647, upper bound: 59.3434162
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3341296, upper bound: 59.3584607
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3584607, upper bound: 59.3341296
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3434162, upper bound: 59.3454647
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3503042, upper bound: 59.3389881
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3389283, upper bound: 59.3540690
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3539974, upper bound: 59.3387342
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3388690, upper bound: 59.3499472
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3458017, upper bound: 59.3435034
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.22
Output dim: 6, lower bound: -59.3343187, upper bound: 59.3585259

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3510182, upper bound: 59.3235408
time: 172.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3486541, upper bound: 59.3266457
time: 150.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3358512, upper bound: 59.3351303
time: 113.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3332533, upper bound: 59.3382045
time: 127.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3423607, upper bound: 59.3282124
time: 159.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3398476, upper bound: 59.3312963
time: 157.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3310662, upper bound: 59.3437839
time: 139.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3283466, upper bound: 59.3466040
time: 117.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3466919, upper bound: 59.3284244
time: 134.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3438422, upper bound: 59.3310818
time: 195.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3314255, upper bound: 59.3400380
time: 140.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3283789, upper bound: 59.3426507
time: 158.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3379513, upper bound: 59.3330990
time: 137.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3349426, upper bound: 59.3357263
time: 163.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -85.1744843, 35.2064743, -85.1744843, 35.2064743, -120.3809586, 120.3809586
1: -48.2351494, 33.1485405, -48.2351494, 33.1485405, -81.3836899, 81.3836823
2: -41.1655655, 28.5826340, -41.1655655, 28.5826340, -69.7481995, 69.7481995
3: -46.5626945, 38.4125671, -46.5626945, 38.4125671, -84.9752655, 84.9752655
4: -53.6668243, 36.7295227, -53.6668243, 36.7295227, -90.3963394, 90.3963318
5: -46.2806854, 40.4751282, -46.2806854, 40.4751282, -86.7558060, 86.7558136
6: -42.0284424, 41.0051041, -42.0284424, 41.0051041, -83.0335464, 83.0335388
7: -54.6045380, 41.1102600, -54.6045380, 41.1102600, -95.7147980, 95.7147980
8: -56.8510666, 48.0971832, -56.8510666, 48.0971832, -104.9482498, 104.9482498
9: -44.0658836, 41.3547745, -44.0658836, 41.3547745, -85.4206543, 85.4206543
10: -65.4405670, 53.0221558, -65.4405670, 53.0221558, -118.4627228, 118.4627228
11: -62.5605507, 40.9191666, -62.5605507, 40.9191666, -103.4797211, 103.4797211
12: -50.5190811, 49.9758148, -50.5190811, 49.9758148, -100.4948807, 100.4948959
13: -64.5225983, 55.0663376, -64.5225983, 55.0663376, -119.5889130, 119.5889359
14: -104.3223495, 24.1945820, -104.3223495, 24.1945820, -128.5169220, 128.5169220
15: -54.5036621, 33.1086884, -54.5036621, 33.1086884, -87.6123428, 87.6123428
16: -68.2884750, 46.5362930, -68.2884750, 46.5362930, -114.8247681, 114.8247681
17: -110.0658264, 44.2136765, -110.0658264, 44.2136765, -154.2794800, 154.2794952
18: -58.8214874, 46.7788277, -58.8214874, 46.7788277, -105.6003113, 105.6003113
19: -46.8095093, 22.8448448, -46.8095093, 22.8448448, -69.6543579, 69.6543579
20: -40.6778488, 32.3519630, -40.6778488, 32.3519630, -73.0298157, 73.0298157
21: -61.4697647, 29.7116356, -61.4697647, 29.7116356, -91.1813812, 91.1813965
22: -67.8623810, 26.6814575, -67.8623810, 26.6814575, -94.5438385, 94.5438385
23: -45.7086029, 33.9955215, -45.7086029, 33.9955215, -79.7041168, 79.7041168
24: -60.3867188, 36.3501892, -60.3867188, 36.3501892, -96.7369080, 96.7369080
25: -47.1769295, 34.1564102, -47.1769295, 34.1564102, -81.3333435, 81.3333282
26: -66.6641998, 52.3930206, -66.6641998, 52.3930206, -119.0572205, 119.0572205
27: -58.9503403, 38.1289673, -58.9503403, 38.1289673, -97.0793076, 97.0792999
28: -44.0236588, 37.3034363, -44.0236588, 37.3034363, -81.3270950, 81.3270950
29: -76.0430069, 29.6796322, -76.0430069, 29.6796322, -105.7226410, 105.7226410
30: -56.4353600, 43.2880516, -56.4353600, 43.2880516, -99.7234039, 99.7234116
31: -61.6114426, 31.8376694, -61.6114426, 31.8376694, -93.4491119, 93.4491119
32: -48.2787437, 38.9406052, -48.2787437, 38.9406052, -87.2193451, 87.2193451
33: -71.9006195, 53.7641220, -71.9006195, 53.7641220, -125.6647339, 125.6647263
34: -59.2021103, 45.3630791, -59.2021103, 45.3630791, -104.5651855, 104.5651855
35: -66.7205811, 46.9450378, -66.7205811, 46.9450378, -113.6656036, 113.6656189
36: -58.9930000, 48.7076073, -58.9930000, 48.7076073, -107.7005997, 107.7005920
37: -76.3787994, 48.9754715, -76.3787994, 48.9754715, -125.3542709, 125.3542633
38: -71.0280151, 57.3801804, -71.0280151, 57.3801804, -128.4082031, 128.4082031
39: -88.3980713, 51.0425186, -88.3980713, 51.0425186, -139.4405823, 139.4405823
40: -59.1572685, 46.7854691, -59.1572685, 46.7854691, -105.9427338, 105.9427338
41: -43.7816582, 42.2595367, -43.7816582, 42.2595367, -86.0411835, 86.0411987
42: -32.8602371, 40.8374023, -32.8602371, 40.8374023, -73.6976318, 73.6976318

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3266305, upper bound: 59.3486413
time: 117.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3234656, upper bound: 59.3509435
time: 122.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 241.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3510182, upper bound: 59.3235408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3486541, upper bound: 59.3266457
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3358512, upper bound: 59.3351303
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3332533, upper bound: 59.3382045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3423607, upper bound: 59.3282124
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3398476, upper bound: 59.3312963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3310662, upper bound: 59.3437839
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3283466, upper bound: 59.3466040
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3466919, upper bound: 59.3284244
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3438422, upper bound: 59.3310818
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3314255, upper bound: 59.3400380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3283789, upper bound: 59.3426507
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3379513, upper bound: 59.3330990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3349426, upper bound: 59.3357263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3266305, upper bound: 59.3486413
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 241.88
Output dim: 6, lower bound: -59.3234656, upper bound: 59.3509435
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3584607, upper bound: 59.3341296
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3434162, upper bound: 59.3454647
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3503042, upper bound: 59.3389881
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3389283, upper bound: 59.3540690
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3539974, upper bound: 59.3387342
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3388690, upper bound: 59.3499472
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3458017, upper bound: 59.3435034
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 241.88
Output dim: 6, lower bound: -59.3343187, upper bound: 59.3585259

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 128.24 + 7122.05 = 7250.28 seconds
