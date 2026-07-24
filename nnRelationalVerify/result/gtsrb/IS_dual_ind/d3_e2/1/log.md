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
execution time: IAR + RelationalAnalysis = 2.76 + 121.66 = 124.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -59.3940217, upper bound: 59.3940217

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3689607
time: 116.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3855864
time: 151.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 267.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 267.78
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3689607
IS_A2, status: Status.UNKNOWN, split count: 1, time: 267.78
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3855864

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -85.0557861, 35.1168404, -85.1350250, 35.1761169, -120.2319031, 120.2518616
1: -48.1317596, 33.0766792, -48.2002487, 33.1245422, -81.2562943, 81.2769318
2: -40.9579506, 28.5468254, -41.0963135, 28.5706215, -69.5285721, 69.6431427
3: -46.3437271, 38.3541679, -46.4896240, 38.3929749, -84.7367020, 84.8437881
4: -53.4221992, 36.6823883, -53.5853081, 36.7136612, -90.1358490, 90.2677002
5: -46.0795021, 40.4159698, -46.2136459, 40.4553604, -86.5348663, 86.6296158
6: -41.9551125, 40.8185425, -42.0037689, 40.9428596, -82.8979645, 82.8223114
7: -54.4617996, 41.0618553, -54.5564499, 41.0941315, -95.5559311, 95.6183014
8: -56.5978088, 48.0389938, -56.7665939, 48.0776215, -104.6754303, 104.8055878
9: -44.0178146, 41.2047958, -44.0496140, 41.3045883, -85.3224030, 85.2544098
10: -65.3575897, 52.6700363, -65.4124451, 52.9047661, -118.2623520, 118.0824814
11: -62.4875488, 40.6713791, -62.5357513, 40.8366394, -103.3241882, 103.2071228
12: -50.4570618, 49.5307198, -50.4983406, 49.8276367, -100.2846985, 100.0290604
13: -64.4455948, 54.8973923, -64.4966431, 55.0090981, -119.4546890, 119.3940353
14: -104.2138062, 23.9228630, -104.2860641, 24.1039753, -128.3177643, 128.2089233
15: -54.3469353, 33.0368729, -54.4514046, 33.0844536, -87.4313889, 87.4882736
16: -68.1854324, 46.3308372, -68.2536621, 46.4657974, -114.6512299, 114.5844879
17: -109.9727783, 43.7467346, -110.0348511, 44.0577850, -154.0305634, 153.7815857
18: -58.7369919, 46.6116638, -58.7931938, 46.7226028, -105.4595947, 105.4048615
19: -46.7444305, 22.7391167, -46.7875214, 22.8095207, -69.5539398, 69.5266342
20: -40.5706558, 32.2950745, -40.6418610, 32.3330650, -72.9037170, 72.9369278
21: -61.3912277, 29.5678577, -61.4433174, 29.6637516, -91.0549774, 91.0111694
22: -67.7601166, 26.5686512, -67.8280487, 26.6438866, -94.4040070, 94.3966980
23: -45.6448822, 33.8659515, -45.6872635, 33.9522781, -79.5971603, 79.5532150
24: -60.2502098, 36.3209190, -60.3409920, 36.3404236, -96.5906296, 96.6619110
25: -47.1064529, 34.0778427, -47.1533279, 34.1296082, -81.2360611, 81.2311707
26: -66.5576477, 52.1978455, -66.6285095, 52.3272476, -118.8848801, 118.8263550
27: -58.7695312, 38.0973053, -58.8898125, 38.1182632, -96.8877945, 96.9871216
28: -43.9489479, 37.2555695, -43.9986153, 37.2873077, -81.2362518, 81.2541809
29: -75.9521484, 29.5229626, -76.0125580, 29.6271839, -105.5793304, 105.5355225
30: -56.3591499, 43.1980286, -56.4097023, 43.2578049, -99.6169586, 99.6077271
31: -61.5189590, 31.7429466, -61.5804710, 31.8053665, -93.3243256, 93.3234177
32: -48.2176285, 38.7799911, -48.2581406, 38.8870621, -87.1046906, 87.0381317
33: -71.7478638, 53.7028770, -71.8496552, 53.7435379, -125.4914017, 125.5525360
34: -59.0820732, 45.2831039, -59.1620674, 45.3361168, -104.4181824, 104.4451752
35: -66.5796204, 46.9040680, -66.6735382, 46.9312325, -113.5108490, 113.5776062
36: -58.8984680, 48.6482773, -58.9612846, 48.6878815, -107.5863419, 107.6095581
37: -76.2586060, 48.7916870, -76.3385468, 48.9140053, -125.1726074, 125.1302338
38: -70.8898468, 57.2528114, -70.9817886, 57.3369026, -128.2267456, 128.2346039
39: -88.2587891, 50.9949722, -88.3511658, 51.0265503, -139.2853241, 139.3461304
40: -58.9897423, 46.7318382, -59.1010399, 46.7673721, -105.7571106, 105.8328781
41: -43.7025871, 42.0897446, -43.7550735, 42.2029152, -85.9055023, 85.8448181
42: -32.8076324, 40.5895462, -32.8425293, 40.7548065, -73.5624313, 73.4320755

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3059511
time: 134.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3562081
time: 120.98 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -85.1973267, 35.1975021, -85.1691742, 35.1968346, -120.3941650, 120.3666687
1: -48.2348557, 33.1464157, -48.2304459, 33.1426430, -81.3774948, 81.3768616
2: -41.1587524, 28.6137924, -41.1587372, 28.5801849, -69.7389374, 69.7725296
3: -46.5562668, 38.4410439, -46.5555458, 38.4086494, -84.9649200, 84.9965897
4: -53.6618614, 36.7826080, -53.6586418, 36.7268829, -90.3887482, 90.4412384
5: -46.2766953, 40.5021744, -46.2745209, 40.4720764, -86.7487717, 86.7766953
6: -42.0693817, 40.9983139, -42.0245323, 40.9981232, -83.0675049, 83.0228424
7: -54.6044273, 41.1107826, -54.5976448, 41.1056519, -95.7100830, 95.7084198
8: -56.8457413, 48.1645775, -56.8423309, 48.0932198, -104.9389572, 105.0068970
9: -44.0753555, 41.3639183, -44.0632782, 41.3491211, -85.4244766, 85.4272003
10: -65.5016479, 53.0154266, -65.4360352, 53.0107117, -118.5123367, 118.4514618
11: -62.6081085, 40.9072456, -62.5547256, 40.9105225, -103.5186310, 103.4619751
12: -50.6389999, 49.9549942, -50.5154419, 49.9612007, -100.6002045, 100.4704361
13: -64.5175781, 55.0746231, -64.5152130, 55.0574379, -119.5750122, 119.5898361
14: -104.3607559, 24.1830635, -104.3159485, 24.1862316, -128.5469818, 128.4990082
15: -54.4619102, 33.1384811, -54.4792786, 33.1043129, -87.5662231, 87.6177521
16: -68.3243561, 46.5032387, -68.2825851, 46.5181274, -114.8424835, 114.7858200
17: -110.2187576, 44.1904869, -110.0610199, 44.1965561, -154.4153137, 154.2515106
18: -58.8488655, 46.7747078, -58.8153687, 46.7725029, -105.6213684, 105.5900574
19: -46.8487396, 22.8455601, -46.8061333, 22.8409481, -69.6896820, 69.6516800
20: -40.6906471, 32.3519516, -40.6724548, 32.3489914, -73.0396347, 73.0244064
21: -61.4924316, 29.7043991, -61.4657249, 29.7067451, -91.1991653, 91.1701202
22: -67.8576889, 26.6872692, -67.8503647, 26.6760216, -94.5337067, 94.5376358
23: -45.7295990, 33.9906006, -45.7056427, 33.9906044, -79.7201996, 79.6962433
24: -60.3925285, 36.3594780, -60.3793221, 36.3485947, -96.7411194, 96.7388000
25: -47.1896439, 34.1569977, -47.1720963, 34.1515656, -81.3412094, 81.3290939
26: -66.6572418, 52.3841171, -66.6501541, 52.3833389, -119.0405579, 119.0342560
27: -58.9542732, 38.1567078, -58.9418526, 38.1264458, -97.0807114, 97.0985565
28: -44.0342712, 37.3059540, -44.0196228, 37.2999649, -81.3342285, 81.3255768
29: -76.0443115, 29.6763515, -76.0338898, 29.6737556, -105.7180557, 105.7102432
30: -56.4416199, 43.2774544, -56.4295349, 43.2769775, -99.7185974, 99.7069855
31: -61.6491089, 31.8347321, -61.6074028, 31.8330917, -93.4822006, 93.4421234
32: -48.3046036, 38.9271660, -48.2754021, 38.9328232, -87.2374268, 87.2025604
33: -71.9095001, 53.7708054, -71.8944168, 53.7584991, -125.6679916, 125.6652222
34: -59.2041779, 45.3616142, -59.1969223, 45.3545914, -104.5587692, 104.5585327
35: -66.7222900, 46.9675217, -66.7143402, 46.9421616, -113.6644440, 113.6818619
36: -58.9907837, 48.7150879, -58.9845123, 48.7050171, -107.6958008, 107.6996002
37: -76.3999939, 48.9723434, -76.3714676, 48.9694633, -125.3694458, 125.3438110
38: -71.0374756, 57.3860893, -71.0201263, 57.3746681, -128.4121399, 128.4062195
39: -88.4116364, 51.0547485, -88.3898010, 51.0386162, -139.4502563, 139.4445496
40: -59.1718636, 46.7867126, -59.1500168, 46.7819595, -105.9538116, 105.9367294
41: -43.8297615, 42.2509079, -43.7778130, 42.2530365, -86.0827942, 86.0287170
42: -32.9505463, 40.8240967, -32.8575287, 40.8282471, -73.7787933, 73.6816254

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3267093
time: 129.09 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3734268
time: 116.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 247.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 247.42
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3059511
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 247.42
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3562081
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 247.42
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3267093
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 247.42
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3734268

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -84.9794769, 35.0509872, -84.9036560, 34.9815331, -119.9609985, 119.9546432
1: -48.0643005, 33.0259628, -47.9986343, 32.9706116, -81.0349121, 81.0245972
2: -40.8406754, 28.5197163, -40.7482300, 28.4786987, -69.3193741, 69.2679443
3: -46.2222176, 38.3084564, -46.1364975, 38.2496414, -84.4718628, 84.4449463
4: -53.2784424, 36.6453438, -53.1638069, 36.6205063, -89.8989487, 89.8091507
5: -45.9829750, 40.3715782, -45.9332008, 40.3179474, -86.3009186, 86.3047791
6: -41.8936310, 40.6968880, -41.8305893, 40.5843124, -82.4779358, 82.5274811
7: -54.3875198, 41.0188675, -54.3341408, 40.9639359, -95.3514557, 95.3530121
8: -56.4390030, 47.9933891, -56.2976532, 47.9250145, -104.3640137, 104.2910461
9: -43.9839973, 41.1000290, -43.9284973, 40.9947586, -84.9787598, 85.0285263
10: -65.2931824, 52.3871803, -65.1313858, 52.0750160, -117.3681793, 117.5185623
11: -62.4344788, 40.4953537, -62.3681946, 40.3229561, -102.7574310, 102.8635483
12: -50.4111824, 49.2691307, -50.2930832, 49.0554886, -99.4666748, 99.5622025
13: -64.3951569, 54.7928200, -64.3533325, 54.6875343, -119.0826874, 119.1461487
14: -104.1460648, 23.7394123, -104.0638199, 23.5598660, -127.7059326, 127.8032303
15: -54.2328606, 32.9824982, -54.1124878, 32.9618378, -87.1947021, 87.0949860
16: -68.1144333, 46.1552124, -68.0414581, 45.9543037, -114.0687408, 114.1966705
17: -109.9184875, 43.4386673, -109.7949524, 43.1430397, -153.0615234, 153.2336121
18: -58.6719170, 46.4946518, -58.5793114, 46.3776512, -105.0495605, 105.0739594
19: -46.6950684, 22.6719284, -46.6208839, 22.6124172, -69.3074875, 69.2928162
20: -40.5037575, 32.2617302, -40.4356003, 32.2357216, -72.7394791, 72.6973267
21: -61.3332176, 29.4819450, -61.2635040, 29.4148293, -90.7480469, 90.7454453
22: -67.6743927, 26.5028553, -67.5700073, 26.4830132, -94.1573868, 94.0728607
23: -45.6013107, 33.7848930, -45.5627556, 33.7157593, -79.3170700, 79.3476410
24: -60.1500359, 36.3018684, -60.0337753, 36.2816772, -96.4317169, 96.3356476
25: -47.0527802, 34.0315895, -46.9850845, 33.9904861, -81.0432587, 81.0166779
26: -66.4795685, 52.0907364, -66.3868713, 52.0082626, -118.4878311, 118.4776001
27: -58.6201897, 38.0738564, -58.4397469, 38.0219536, -96.6421432, 96.5136032
28: -43.8827972, 37.2290955, -43.7962570, 37.2012787, -81.0840759, 81.0253448
29: -75.8779449, 29.4497280, -75.7844925, 29.4130020, -105.2909470, 105.2342224
30: -56.3066139, 43.1448288, -56.2563057, 43.0982552, -99.4048691, 99.4011230
31: -61.4513931, 31.6679420, -61.3709183, 31.5856476, -93.0370407, 93.0388641
32: -48.1682167, 38.6994553, -48.0995064, 38.6527481, -86.8209610, 86.7989655
33: -71.6273193, 53.6643867, -71.4906464, 53.6332016, -125.2605209, 125.1550293
34: -58.9824753, 45.2328033, -58.8660240, 45.1786766, -104.1611481, 104.0988159
35: -66.4490509, 46.8713150, -66.2893066, 46.7930641, -113.2421036, 113.1606216
36: -58.7922096, 48.6145325, -58.6493645, 48.5825729, -107.3747787, 107.2639008
37: -76.1564865, 48.7249069, -76.0269089, 48.7154236, -124.8719025, 124.7518158
38: -70.7610168, 57.2005005, -70.6060562, 57.1731796, -127.9341965, 127.8065567
39: -88.1432800, 50.9668808, -88.0022888, 50.9561462, -139.0994263, 138.9691772
40: -58.8539734, 46.7030373, -58.6881256, 46.6888885, -105.5428467, 105.3911591
41: -43.6371918, 41.9983253, -43.5547295, 41.9351807, -85.5723724, 85.5530548
42: -32.7697906, 40.4219856, -32.6830215, 40.2619553, -73.0317383, 73.1050110

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1934347, upper bound: 59.2854755
time: 123.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2954080, upper bound: 59.3027970
time: 133.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -85.0555191, 35.1164093, -85.1263733, 35.1625748, -120.2180939, 120.2427826
1: -48.1315346, 33.0764618, -48.1930389, 33.1176796, -81.2492065, 81.2695007
2: -40.9576607, 28.5466805, -41.0869484, 28.5659962, -69.5236588, 69.6336288
3: -46.3433533, 38.3539467, -46.4782104, 38.3859291, -84.7292786, 84.8321533
4: -53.4216537, 36.6822243, -53.5676613, 36.7088165, -90.1304550, 90.2498856
5: -46.0792046, 40.4157677, -46.2040024, 40.4490967, -86.5283051, 86.6197662
6: -41.9548874, 40.8180275, -41.9973679, 40.9300842, -82.8849716, 82.8153992
7: -54.4615250, 41.0615349, -54.5479813, 41.0838013, -95.5453262, 95.6095123
8: -56.5973473, 48.0387650, -56.7526169, 48.0708389, -104.6681824, 104.7913818
9: -44.0176468, 41.2044907, -44.0444107, 41.2950668, -85.3127136, 85.2489014
10: -65.3573456, 52.6692276, -65.4048462, 52.8790894, -118.2364349, 118.0740738
11: -62.4873009, 40.6708565, -62.5286751, 40.8202667, -103.3075714, 103.1995316
12: -50.4568443, 49.5300217, -50.4919891, 49.8049622, -100.2617950, 100.0220032
13: -64.4451294, 54.8970299, -64.4819641, 54.9978600, -119.4429932, 119.3789978
14: -104.2134552, 23.9224091, -104.2748718, 24.0893841, -128.3028412, 128.1972809
15: -54.3459930, 33.0366402, -54.4252014, 33.0783157, -87.4243088, 87.4618378
16: -68.1851120, 46.3302650, -68.2440186, 46.4473724, -114.6324844, 114.5742722
17: -109.9725037, 43.7459602, -110.0260162, 44.0334320, -154.0059357, 153.7719727
18: -58.7366791, 46.6112900, -58.7831879, 46.7110214, -105.4476929, 105.3944778
19: -46.7442360, 22.7388916, -46.7814484, 22.8027821, -69.5470123, 69.5203400
20: -40.5704117, 32.2949219, -40.6349106, 32.3283081, -72.8987198, 72.9298325
21: -61.3909988, 29.5675850, -61.4364777, 29.6551666, -91.0461578, 91.0040588
22: -67.7593842, 26.5684147, -67.8049164, 26.6367512, -94.3961334, 94.3733292
23: -45.6447296, 33.8655815, -45.6823044, 33.9423828, -79.5871124, 79.5478821
24: -60.2498169, 36.3208237, -60.3283577, 36.3370590, -96.5868607, 96.6491776
25: -47.1061783, 34.0775528, -47.1445389, 34.1213264, -81.2275085, 81.2220917
26: -66.5568542, 52.1974030, -66.6038513, 52.3135910, -118.8704376, 118.8012543
27: -58.7690544, 38.0971489, -58.8751945, 38.1133270, -96.8823853, 96.9723358
28: -43.9487228, 37.2554398, -43.9913940, 37.2834587, -81.2321777, 81.2468262
29: -75.9517441, 29.5227337, -75.9993210, 29.6198502, -105.5715790, 105.5220490
30: -56.3588715, 43.1972809, -56.4018440, 43.2366714, -99.5955429, 99.5991211
31: -61.5187378, 31.7426567, -61.5734444, 31.7963867, -93.3151169, 93.3161011
32: -48.2174301, 38.7796135, -48.2523422, 38.8748703, -87.0923004, 87.0319519
33: -71.7474442, 53.7027054, -71.8364716, 53.7386551, -125.4860992, 125.5391693
34: -59.0817070, 45.2827110, -59.1503525, 45.3233109, -104.4050140, 104.4330597
35: -66.5791397, 46.9039230, -66.6588898, 46.9272804, -113.5064087, 113.5628128
36: -58.8980751, 48.6481323, -58.9486198, 48.6837997, -107.5818787, 107.5967484
37: -76.2579803, 48.7913742, -76.3184509, 48.9047470, -125.1627274, 125.1098175
38: -70.8893890, 57.2523766, -70.9671707, 57.3238182, -128.2131958, 128.2195435
39: -88.2577972, 50.9948235, -88.3194733, 51.0221939, -139.2799683, 139.3142853
40: -58.9892998, 46.7316933, -59.0879745, 46.7627716, -105.7520752, 105.8196716
41: -43.7023468, 42.0894318, -43.7481003, 42.1929169, -85.8952637, 85.8375244
42: -32.8074913, 40.5890121, -32.8382721, 40.7382240, -73.5457153, 73.4272842

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2779264
time: 138.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3260866, upper bound: 59.3562081
time: 113.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -85.1216431, 35.1311531, -84.9381561, 35.0016975, -120.1233368, 120.0693054
1: -48.1667137, 33.0955849, -48.0280800, 32.9886398, -81.1553497, 81.1236649
2: -41.0411110, 28.5868187, -40.8102303, 28.4882889, -69.5293961, 69.3970490
3: -46.4341965, 38.3959694, -46.2013245, 38.2653885, -84.6995850, 84.5972900
4: -53.5170441, 36.7457733, -53.2363281, 36.6334114, -90.1504440, 89.9821014
5: -46.1798134, 40.4579849, -45.9933701, 40.3346710, -86.5144806, 86.4513550
6: -42.0077362, 40.8762512, -41.8507957, 40.6388168, -82.6465530, 82.7270508
7: -54.5289650, 41.0678711, -54.3739014, 40.9755669, -95.5045319, 95.4417725
8: -56.6863022, 48.1191788, -56.3725052, 47.9407959, -104.6270828, 104.4916840
9: -44.0417137, 41.2584152, -43.9421234, 41.0385208, -85.0802307, 85.2005310
10: -65.4374313, 52.7321472, -65.1549377, 52.1803665, -117.6177979, 117.8870850
11: -62.5547981, 40.7310791, -62.3864479, 40.3964767, -102.9512787, 103.1175232
12: -50.5930481, 49.6929932, -50.3098221, 49.1884537, -99.7815018, 100.0028000
13: -64.4667969, 54.9679756, -64.3712921, 54.7321777, -119.1989746, 119.3392563
14: -104.2932739, 23.9992390, -104.0937042, 23.6414433, -127.9347153, 128.0929413
15: -54.3472519, 33.0837784, -54.1431465, 32.9812241, -87.3284683, 87.2269287
16: -68.2538757, 46.3271408, -68.0707703, 46.0016060, -114.2554779, 114.3979111
17: -110.1645660, 43.8813057, -109.8211288, 43.2801437, -153.4447021, 153.7024384
18: -58.7838783, 46.6570816, -58.6014671, 46.4267502, -105.2106323, 105.2585449
19: -46.7998161, 22.7780991, -46.6395264, 22.6433716, -69.4431839, 69.4176254
20: -40.6229095, 32.3186722, -40.4650269, 32.2517471, -72.8746567, 72.7836914
21: -61.4342346, 29.6184044, -61.2854767, 29.4576874, -90.8919220, 90.9038773
22: -67.7727051, 26.6220417, -67.5929565, 26.5154037, -94.2881012, 94.2149963
23: -45.6860580, 33.9083557, -45.5811310, 33.7530746, -79.4391327, 79.4894867
24: -60.2920952, 36.3403969, -60.0717278, 36.2896004, -96.5816956, 96.4121246
25: -47.1355553, 34.1094627, -47.0036469, 34.0116806, -81.1472321, 81.1131134
26: -66.5791779, 52.2767487, -66.4081802, 52.0638580, -118.6430359, 118.6849289
27: -58.8041687, 38.1333122, -58.4907303, 38.0300102, -96.8341675, 96.6240387
28: -43.9678841, 37.2792358, -43.8169785, 37.2139168, -81.1817932, 81.0962143
29: -75.9701309, 29.6024437, -75.8055954, 29.4582367, -105.4283600, 105.4080353
30: -56.3887367, 43.2238350, -56.2756653, 43.1172447, -99.5059814, 99.4994888
31: -61.5815086, 31.7594090, -61.3979416, 31.6128769, -93.1943817, 93.1573486
32: -48.2550583, 38.8465729, -48.1162491, 38.6984024, -86.9534607, 86.9628143
33: -71.7888489, 53.7320900, -71.5350037, 53.6479492, -125.4367828, 125.2670898
34: -59.1042747, 45.3108292, -58.9003525, 45.1966286, -104.3008881, 104.2111816
35: -66.5915985, 46.9352493, -66.3298035, 46.8039207, -113.3955154, 113.2650528
36: -58.8844643, 48.6813965, -58.6723671, 48.6001091, -107.4845734, 107.3537598
37: -76.2983627, 48.9046822, -76.0597534, 48.7700806, -125.0684357, 124.9644318
38: -70.9082108, 57.3324089, -70.6436157, 57.2089462, -128.1171570, 127.9760208
39: -88.2959824, 51.0265350, -88.0404053, 50.9681740, -139.2641296, 139.0669250
40: -59.0351410, 46.7575836, -58.7356224, 46.7031631, -105.7382965, 105.4932022
41: -43.7645683, 42.1593018, -43.5771141, 41.9850769, -85.7496414, 85.7364120
42: -32.9126663, 40.6562576, -32.6976700, 40.3350143, -73.2476807, 73.3539276

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1934347, upper bound: 59.3063394
time: 165.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2963425, upper bound: 59.3235653
time: 127.43 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -85.1970367, 35.1970673, -85.1605072, 35.1833115, -120.3803482, 120.3575745
1: -48.2346153, 33.1461945, -48.2231445, 33.1357727, -81.3703918, 81.3693390
2: -41.1584587, 28.6136303, -41.1493301, 28.5755386, -69.7339935, 69.7629547
3: -46.5559082, 38.4408188, -46.5440826, 38.4015770, -84.9574814, 84.9849014
4: -53.6613121, 36.7824440, -53.6409149, 36.7220345, -90.3833466, 90.4233551
5: -46.2763977, 40.5019684, -46.2648010, 40.4658241, -86.7422180, 86.7667694
6: -42.0691681, 40.9978409, -42.0181046, 40.9853363, -83.0545044, 83.0159454
7: -54.6041603, 41.1104469, -54.5890732, 41.0953064, -95.6994629, 95.6995163
8: -56.8452950, 48.1643600, -56.8282928, 48.0863876, -104.9316788, 104.9926529
9: -44.0751877, 41.3636169, -44.0580444, 41.3395844, -85.4147644, 85.4216614
10: -65.5014038, 53.0146103, -65.4283752, 52.9849701, -118.4863739, 118.4429855
11: -62.6078568, 40.9067307, -62.5476913, 40.8941269, -103.5019836, 103.4544220
12: -50.6387939, 49.9542923, -50.5090790, 49.9384613, -100.5772476, 100.4633713
13: -64.5171280, 55.0742760, -64.5004272, 55.0459976, -119.5631256, 119.5746918
14: -104.3603973, 24.1825981, -104.3046951, 24.1715927, -128.5319824, 128.4872894
15: -54.4609756, 33.1382790, -54.4542465, 33.0981941, -87.5591736, 87.5925140
16: -68.3240509, 46.5026474, -68.2729340, 46.4996071, -114.8236542, 114.7755737
17: -110.2184753, 44.1896973, -110.0521774, 44.1721649, -154.3906097, 154.2418518
18: -58.8485413, 46.7743301, -58.8052483, 46.7608910, -105.6094284, 105.5795746
19: -46.8485413, 22.8453350, -46.8000107, 22.8341732, -69.6827087, 69.6453400
20: -40.6904106, 32.3517990, -40.6655960, 32.3442192, -73.0346222, 73.0173950
21: -61.4922180, 29.7041283, -61.4588623, 29.6981316, -91.1903458, 91.1629944
22: -67.8569183, 26.6870327, -67.8268433, 26.6688786, -94.5257950, 94.5138779
23: -45.7294426, 33.9902153, -45.7006645, 33.9807892, -79.7102280, 79.6908798
24: -60.3921204, 36.3593750, -60.3666420, 36.3452377, -96.7373581, 96.7260132
25: -47.1893692, 34.1567421, -47.1632805, 34.1435471, -81.3329163, 81.3200226
26: -66.6564636, 52.3836594, -66.6254349, 52.3695221, -119.0259857, 119.0090942
27: -58.9537964, 38.1565323, -58.9271774, 38.1214828, -97.0752792, 97.0837097
28: -44.0340424, 37.3058243, -44.0123825, 37.2960663, -81.3301086, 81.3182068
29: -76.0438843, 29.6761150, -76.0206146, 29.6662903, -105.7101746, 105.6967316
30: -56.4413452, 43.2767181, -56.4217033, 43.2558594, -99.6972046, 99.6984177
31: -61.6488800, 31.8344402, -61.6003380, 31.8241005, -93.4729691, 93.4347763
32: -48.3044090, 38.9267845, -48.2695999, 38.9205475, -87.2249527, 87.1963806
33: -71.9090805, 53.7706451, -71.8812256, 53.7536011, -125.6626663, 125.6518555
34: -59.2037888, 45.3612061, -59.1851692, 45.3415909, -104.5453796, 104.5463715
35: -66.7218246, 46.9673920, -66.6996613, 46.9381866, -113.6600113, 113.6670456
36: -58.9903717, 48.7149506, -58.9717865, 48.7009048, -107.6912766, 107.6867371
37: -76.3993530, 48.9720421, -76.3513107, 48.9601021, -125.3594513, 125.3233490
38: -71.0370331, 57.3856659, -71.0054474, 57.3615074, -128.3985443, 128.3911133
39: -88.4106598, 51.0546036, -88.3580170, 51.0342865, -139.4449463, 139.4126282
40: -59.1714363, 46.7865601, -59.1369591, 46.7773170, -105.9487305, 105.9235229
41: -43.8295326, 42.2505875, -43.7708168, 42.2429657, -86.0724945, 86.0214081
42: -32.9503975, 40.8235741, -32.8532639, 40.8116150, -73.7620087, 73.6768341

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3267094, upper bound: 59.2994946
time: 108.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2994946
time: 105.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 216.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.1934347, upper bound: 59.2854755
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.2954080, upper bound: 59.3027970
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2779264
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.3260866, upper bound: 59.3562081
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.1934347, upper bound: 59.3063394
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.2963425, upper bound: 59.3235653
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.3267094, upper bound: 59.2994946
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 216.12
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2994946

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -84.8261719, 34.9898529, -84.8590698, 34.9582863, -119.7844543, 119.8489151
1: -47.9857407, 32.8702240, -47.9861145, 32.9184914, -80.9042206, 80.8563385
2: -40.7703896, 28.3488197, -40.7368393, 28.4213943, -69.1917877, 69.0856628
3: -46.1367683, 38.0367355, -46.1274796, 38.1595917, -84.2963486, 84.1642151
4: -53.1787682, 36.4082565, -53.1514359, 36.5406876, -89.7194519, 89.5596924
5: -45.9154205, 40.2006531, -45.9221535, 40.2605209, -86.1759338, 86.1228027
6: -41.6666489, 40.6012192, -41.7543793, 40.5690842, -82.2357330, 82.3555984
7: -54.3193321, 40.8195343, -54.3232536, 40.8962212, -95.2155533, 95.1427917
8: -56.3256531, 47.6329231, -56.2862625, 47.8048325, -104.1304855, 103.9191818
9: -43.9338837, 41.0429993, -43.9130211, 40.9760666, -84.9099503, 84.9560242
10: -65.1591187, 52.3086739, -65.0900116, 52.0595779, -117.2186966, 117.3986740
11: -62.2848320, 40.4246063, -62.3199883, 40.3073425, -102.5921783, 102.7445908
12: -50.2023697, 49.1687546, -50.2218170, 49.0377693, -99.2401428, 99.3905640
13: -64.2952576, 54.5782661, -64.3338242, 54.6175079, -118.9127655, 118.9120941
14: -104.0151978, 23.6796207, -104.0239792, 23.5336609, -127.5488434, 127.7035980
15: -54.1328430, 32.8221588, -54.0953712, 32.9079857, -87.0408249, 86.9175262
16: -67.8967285, 46.0658951, -67.9710236, 45.9432030, -113.8399124, 114.0369186
17: -109.8016510, 43.3565788, -109.7588959, 43.1162071, -152.9178619, 153.1154785
18: -58.4003220, 46.3898849, -58.4914703, 46.3659286, -104.7662506, 104.8813477
19: -46.5241928, 22.6164188, -46.5646362, 22.6052704, -69.1294556, 69.1810532
20: -40.4433823, 32.2080345, -40.4154358, 32.2210007, -72.6643829, 72.6234665
21: -61.2132149, 29.4341316, -61.2249756, 29.4019279, -90.6151428, 90.6591034
22: -67.5813522, 26.4356976, -67.5410233, 26.4586830, -94.0400314, 93.9767075
23: -45.4826736, 33.7188606, -45.5238190, 33.7033768, -79.1860504, 79.2426758
24: -59.9939690, 36.2546196, -59.9838982, 36.2703400, -96.2643127, 96.2385178
25: -46.9275665, 33.9937439, -46.9443932, 33.9796371, -80.9072037, 80.9381332
26: -66.3880692, 52.0270233, -66.3510666, 51.9897537, -118.3778076, 118.3780823
27: -58.5371895, 38.0177307, -58.4142418, 38.0045319, -96.5417175, 96.4319763
28: -43.7970505, 37.1669731, -43.7673454, 37.1861038, -80.9831543, 80.9343109
29: -75.7708054, 29.3975430, -75.7527390, 29.3956718, -105.1664734, 105.1502838
30: -56.2033043, 43.0576172, -56.2223663, 43.0759697, -99.2792587, 99.2799835
31: -61.1624413, 31.5742493, -61.2765884, 31.5755539, -92.7379913, 92.8508377
32: -48.0452271, 38.6283188, -48.0582962, 38.6367493, -86.6819687, 86.6866150
33: -71.5015411, 53.6099739, -71.4492493, 53.6218643, -125.1234055, 125.0592194
34: -58.8429718, 45.1625023, -58.8200798, 45.1662140, -104.0091858, 103.9825745
35: -66.3330536, 46.8154030, -66.2513428, 46.7827339, -113.1157837, 113.0667419
36: -58.7324867, 48.5876350, -58.6283836, 48.5752716, -107.3077469, 107.2160187
37: -75.8499908, 48.6469727, -75.9255829, 48.7062378, -124.5562286, 124.5725555
38: -70.6610260, 57.1362801, -70.5742035, 57.1580887, -127.8191147, 127.7104721
39: -88.0190277, 50.9263382, -87.9627838, 50.9432869, -138.9623108, 138.8891296
40: -58.6225395, 46.6220360, -58.6129532, 46.6789703, -105.3015137, 105.2349854
41: -43.4781494, 41.9286423, -43.5008926, 41.9232788, -85.4014206, 85.4295349
42: -32.7020111, 40.3635788, -32.6599121, 40.2482910, -72.9503021, 73.0234909

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1757885, upper bound: 59.2854755
time: 116.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1757885, upper bound: 59.2854755
time: 118.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -84.9739532, 35.0478477, -84.9035492, 34.9814720, -119.9554214, 119.9514008
1: -48.0622482, 33.0211487, -47.9985886, 32.9705200, -81.0327682, 81.0197372
2: -40.8387642, 28.5146618, -40.7481995, 28.4786034, -69.3173676, 69.2628632
3: -46.2202454, 38.3006096, -46.1364594, 38.2494888, -84.4697342, 84.4370651
4: -53.2765121, 36.6383591, -53.1637573, 36.6203651, -89.8968811, 89.8021164
5: -45.9806595, 40.3661194, -45.9331589, 40.3178329, -86.2984924, 86.2992706
6: -41.8855362, 40.6946487, -41.8304291, 40.5842667, -82.4698029, 82.5250778
7: -54.3850784, 41.0125771, -54.3340912, 40.9638214, -95.3488922, 95.3466644
8: -56.4374237, 47.9828072, -56.2976151, 47.9248047, -104.3622131, 104.2804184
9: -43.9817848, 41.0978088, -43.9284592, 40.9947128, -84.9764862, 85.0262604
10: -65.2857513, 52.3846893, -65.1312408, 52.0749626, -117.3607025, 117.5159302
11: -62.4255905, 40.4930801, -62.3680153, 40.3229141, -102.7485046, 102.8610992
12: -50.4036179, 49.2664413, -50.2929420, 49.0554352, -99.4590530, 99.5593872
13: -64.3912048, 54.7892380, -64.3532410, 54.6874695, -119.0786743, 119.1424713
14: -104.1371918, 23.7361469, -104.0636520, 23.5598125, -127.6970062, 127.7997971
15: -54.2295532, 32.9696960, -54.1124191, 32.9615898, -87.1911392, 87.0821075
16: -68.1047134, 46.1526794, -68.0412292, 45.9542656, -114.0589752, 114.1939087
17: -109.9136963, 43.4349518, -109.7948456, 43.1429787, -153.0566559, 153.2297974
18: -58.6641541, 46.4925537, -58.5791550, 46.3776245, -105.0417786, 105.0717010
19: -46.6912994, 22.6704979, -46.6208115, 22.6123886, -69.3036804, 69.2913055
20: -40.5007401, 32.2600288, -40.4355431, 32.2356834, -72.7364197, 72.6955719
21: -61.3275757, 29.4802151, -61.2633934, 29.4147949, -90.7423706, 90.7436066
22: -67.6698608, 26.4901657, -67.5699158, 26.4827557, -94.1526184, 94.0600815
23: -45.5932770, 33.7831573, -45.5626030, 33.7157249, -79.3089981, 79.3457642
24: -60.1438141, 36.2997055, -60.0336456, 36.2816391, -96.4254532, 96.3333511
25: -47.0476418, 34.0295639, -46.9849777, 33.9904404, -81.0380859, 81.0145416
26: -66.4745026, 52.0876808, -66.3867722, 52.0082092, -118.4826965, 118.4744568
27: -58.6172447, 38.0716171, -58.4397049, 38.0219040, -96.6391449, 96.5113220
28: -43.8785553, 37.2268600, -43.7961731, 37.2012405, -81.0797958, 81.0230331
29: -75.8731079, 29.4414673, -75.7843933, 29.4128399, -105.2859497, 105.2258606
30: -56.3030777, 43.1420059, -56.2562180, 43.0981979, -99.4012680, 99.3982239
31: -61.4432297, 31.6654129, -61.3707275, 31.5855865, -93.0288162, 93.0361404
32: -48.1635246, 38.6977654, -48.0994148, 38.6527100, -86.8162384, 86.7971802
33: -71.6227417, 53.6623611, -71.4905624, 53.6331711, -125.2558899, 125.1529236
34: -58.9780922, 45.2303314, -58.8659325, 45.1786232, -104.1567154, 104.0962677
35: -66.4453735, 46.8697968, -66.2892303, 46.7930336, -113.2384033, 113.1590271
36: -58.7889748, 48.6132889, -58.6493111, 48.5825539, -107.3715210, 107.2626038
37: -76.1468353, 48.7233658, -76.0267258, 48.7153931, -124.8622131, 124.7500916
38: -70.7567596, 57.1982841, -70.6059799, 57.1731339, -127.9298935, 127.8042603
39: -88.1389389, 50.9647827, -88.0022125, 50.9561005, -139.0950317, 138.9669800
40: -58.8466263, 46.7010117, -58.6879845, 46.6888504, -105.5354767, 105.3889923
41: -43.6312027, 41.9966965, -43.5546112, 41.9351501, -85.5663528, 85.5513077
42: -32.7640305, 40.4197769, -32.6829147, 40.2619171, -73.0259476, 73.1026917

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782919, upper bound: 59.2005947
time: 118.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782919, upper bound: 59.2005947
time: 105.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -84.8241501, 34.9229584, -85.1263733, 35.1625748, -119.9867249, 120.0493240
1: -47.9311905, 32.9230080, -48.1930389, 33.1176796, -81.0488739, 81.1160431
2: -40.6110306, 28.4548149, -41.0869484, 28.5659962, -69.1770172, 69.5417633
3: -45.9924698, 38.2108536, -46.4782104, 38.3859291, -84.3784027, 84.6890640
4: -53.0060005, 36.5896759, -53.5676613, 36.7088165, -89.7148132, 90.1573334
5: -45.8006821, 40.2784729, -46.2040024, 40.4490967, -86.2497787, 86.4824753
6: -41.7823563, 40.4625549, -41.9973679, 40.9300842, -82.7124405, 82.4599228
7: -54.2408981, 40.9318047, -54.5479813, 41.0838013, -95.3246841, 95.4797821
8: -56.1304245, 47.8857422, -56.7526169, 48.0708389, -104.2012482, 104.6383514
9: -43.8967590, 40.8964310, -44.0444107, 41.2950668, -85.1918182, 84.9408417
10: -65.0765228, 51.8418159, -65.4048462, 52.8790894, -117.9556122, 117.2466583
11: -62.3197365, 40.1588440, -62.5286751, 40.8202667, -103.1399994, 102.6875153
12: -50.2522316, 48.7603111, -50.4919891, 49.8049622, -100.0571747, 99.2522888
13: -64.3032532, 54.5801353, -64.4819641, 54.9978600, -119.3011169, 119.0621033
14: -103.9917450, 23.3803883, -104.2748718, 24.0893841, -128.0811157, 127.6552582
15: -54.0091629, 32.9146385, -54.4252014, 33.0783157, -87.0874786, 87.3398438
16: -67.9720230, 45.8283119, -68.2440186, 46.4473724, -114.4193878, 114.0723267
17: -109.7332535, 42.8351440, -110.0260162, 44.0334320, -153.7666931, 152.8611603
18: -58.5245590, 46.2681351, -58.7831879, 46.7110214, -105.2355804, 105.0513229
19: -46.5775185, 22.5427818, -46.7814484, 22.8027821, -69.3802948, 69.3242340
20: -40.3663864, 32.1977615, -40.6349106, 32.3283081, -72.6946945, 72.8326721
21: -61.2118988, 29.3194790, -61.4364777, 29.6551666, -90.8670654, 90.7559586
22: -67.5016251, 26.4066086, -67.8049164, 26.6367512, -94.1383667, 94.2115250
23: -45.5204926, 33.6325378, -45.6823044, 33.9423828, -79.4628754, 79.3148346
24: -59.9444046, 36.2626343, -60.3283577, 36.3370590, -96.2814560, 96.5909882
25: -46.9389000, 33.9403152, -47.1445389, 34.1213264, -81.0602188, 81.0848541
26: -66.3168869, 51.8806992, -66.6038513, 52.3135910, -118.6304779, 118.4845505
27: -58.3216515, 38.0012093, -58.8751945, 38.1133270, -96.4349670, 96.8764038
28: -43.7471390, 37.1697693, -43.9913940, 37.2834587, -81.0305939, 81.1611633
29: -75.7247009, 29.3094940, -75.9993210, 29.6198502, -105.3445435, 105.3088150
30: -56.2060699, 43.0402374, -56.4018440, 43.2366714, -99.4427414, 99.4420776
31: -61.3092575, 31.5240059, -61.5734444, 31.7963867, -93.1056442, 93.0974503
32: -48.0593872, 38.5463104, -48.2523422, 38.8748703, -86.9342575, 86.7986526
33: -71.3897858, 53.5930901, -71.8364716, 53.7386551, -125.1284180, 125.4295425
34: -58.7872200, 45.1277122, -59.1503525, 45.3233109, -104.1105347, 104.2780609
35: -66.1961212, 46.7654572, -66.6588898, 46.9272804, -113.1233978, 113.4243469
36: -58.5872154, 48.5420685, -58.9486198, 48.6837997, -107.2710114, 107.4906921
37: -75.9474792, 48.5945206, -76.3184509, 48.9047470, -124.8522263, 124.9129639
38: -70.5161743, 57.0911331, -70.9671707, 57.3238182, -127.8399963, 128.0583038
39: -87.9106445, 50.9248657, -88.3194733, 51.0221939, -138.9328308, 139.2443237
40: -58.5789948, 46.6539536, -59.0879745, 46.7627716, -105.3417587, 105.7419281
41: -43.5026207, 41.8226929, -43.7481003, 42.1929169, -85.6955414, 85.5707932
42: -32.6482239, 40.0978050, -32.8382721, 40.7382240, -73.3864441, 72.9360733

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782170, upper bound: 59.1729354
time: 181.64 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782170, upper bound: 59.1729354
time: 205.19 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -85.0471191, 35.1029396, -85.1263733, 35.1625748, -120.2096939, 120.2293091
1: -48.1245193, 33.0697517, -48.1930389, 33.1176796, -81.2422028, 81.2627869
2: -40.9486084, 28.5421371, -41.0869484, 28.5659962, -69.5146027, 69.6290894
3: -46.3322067, 38.3471031, -46.4782104, 38.3859291, -84.7181396, 84.8253098
4: -53.4046860, 36.6774788, -53.5676613, 36.7088165, -90.1135025, 90.2451401
5: -46.0699348, 40.4096298, -46.2040024, 40.4490967, -86.5190277, 86.6136322
6: -41.9484062, 40.8056107, -41.9973679, 40.9300842, -82.8784943, 82.8029785
7: -54.4531288, 41.0515518, -54.5479813, 41.0838013, -95.5369263, 95.5995331
8: -56.5836830, 48.0321045, -56.7526169, 48.0708389, -104.6545105, 104.7847137
9: -44.0126266, 41.1951370, -44.0444107, 41.2950668, -85.3076935, 85.2395477
10: -65.3498840, 52.6444054, -65.4048462, 52.8790894, -118.2289734, 118.0492554
11: -62.4796715, 40.6550713, -62.5286751, 40.8202667, -103.2999420, 103.1837463
12: -50.4505501, 49.5081787, -50.4919891, 49.8049622, -100.2555084, 100.0001678
13: -64.4307251, 54.8860207, -64.4819641, 54.9978600, -119.4285889, 119.3679810
14: -104.2025681, 23.9082985, -104.2748718, 24.0893841, -128.2919464, 128.1831665
15: -54.3206635, 33.0303040, -54.4252014, 33.0783157, -87.3989792, 87.4555054
16: -68.1756134, 46.3123550, -68.2440186, 46.4473724, -114.6229858, 114.5563583
17: -109.9639511, 43.7221718, -110.0260162, 44.0334320, -153.9973755, 153.7481842
18: -58.7272148, 46.5999413, -58.7831879, 46.7110214, -105.4382324, 105.3831253
19: -46.7383575, 22.7322865, -46.7814484, 22.8027821, -69.5411377, 69.5137329
20: -40.5632248, 32.2903137, -40.6349106, 32.3283081, -72.8915253, 72.9252243
21: -61.3842354, 29.5593033, -61.4364777, 29.6551666, -91.0393982, 90.9957733
22: -67.7370148, 26.5610218, -67.8049164, 26.6367512, -94.3737640, 94.3659363
23: -45.6399155, 33.8556557, -45.6823044, 33.9423828, -79.5822983, 79.5379486
24: -60.2376480, 36.3175049, -60.3283577, 36.3370590, -96.5747070, 96.6458588
25: -47.0976143, 34.0692749, -47.1445389, 34.1213264, -81.2189407, 81.2138062
26: -66.5329971, 52.1842804, -66.6038513, 52.3135910, -118.8465729, 118.7881241
27: -58.7547340, 38.0922775, -58.8751945, 38.1133270, -96.8680420, 96.9674683
28: -43.9417496, 37.2515755, -43.9913940, 37.2834587, -81.2252045, 81.2429657
29: -75.9389648, 29.5155621, -75.9993210, 29.6198502, -105.5588150, 105.5148773
30: -56.3508301, 43.1765938, -56.4018440, 43.2366714, -99.5874939, 99.5784378
31: -61.5118713, 31.7338657, -61.5734444, 31.7963867, -93.3082504, 93.3072968
32: -48.2115402, 38.7678528, -48.2523422, 38.8748703, -87.0864105, 87.0201950
33: -71.7346344, 53.6976776, -71.8364716, 53.7386551, -125.4732819, 125.5341415
34: -59.0703201, 45.2706871, -59.1503525, 45.3233109, -104.3936310, 104.4210358
35: -66.5649185, 46.8999138, -66.6588898, 46.9272804, -113.4921875, 113.5588074
36: -58.8857803, 48.6441422, -58.9486198, 48.6837997, -107.5695801, 107.5927582
37: -76.2385406, 48.7821884, -76.3184509, 48.9047470, -125.1432877, 125.1006393
38: -70.8751144, 57.2394943, -70.9671707, 57.3238182, -128.1989136, 128.2066650
39: -88.2270966, 50.9904556, -88.3194733, 51.0221939, -139.2492828, 139.3099213
40: -58.9762650, 46.7271042, -59.0879745, 46.7627716, -105.7390366, 105.8150787
41: -43.6954193, 42.0797577, -43.7481003, 42.1929169, -85.8883362, 85.8278580
42: -32.8031769, 40.5729675, -32.8382721, 40.7382240, -73.5413971, 73.4112396

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782170, upper bound: 59.2507654
time: 93.15 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2782170, upper bound: 59.3530870
time: 146.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -84.9683762, 35.0700378, -84.8935852, 34.9784622, -119.9468384, 119.9636230
1: -48.0881500, 32.9399338, -48.0155563, 32.9365463, -81.0246964, 80.9554901
2: -40.9708557, 28.4159374, -40.7988548, 28.4309845, -69.4018402, 69.2147903
3: -46.3487816, 38.1242447, -46.1923141, 38.1753464, -84.5241241, 84.3165588
4: -53.4174080, 36.5086555, -53.2239609, 36.5535851, -89.9709930, 89.7326050
5: -46.1123123, 40.2870560, -45.9823151, 40.2772331, -86.3895416, 86.2693710
6: -41.7807770, 40.7805824, -41.7745857, 40.6235847, -82.4043427, 82.5551682
7: -54.4607811, 40.8685532, -54.3630257, 40.9078407, -95.3686142, 95.2315750
8: -56.5730057, 47.7586670, -56.3611298, 47.8206177, -104.3936234, 104.1197968
9: -43.9915352, 41.2014008, -43.9266129, 41.0198097, -85.0113373, 85.1280136
10: -65.3033905, 52.6536560, -65.1135330, 52.1649094, -117.4682922, 117.7671890
11: -62.4050980, 40.6603470, -62.3382149, 40.3808975, -102.7859955, 102.9985504
12: -50.3842087, 49.5926399, -50.2385445, 49.1707344, -99.5549469, 99.8311844
13: -64.3668365, 54.7535553, -64.3517914, 54.6622124, -119.0290527, 119.1053467
14: -104.1622849, 23.9392757, -104.0538635, 23.6152039, -127.7774811, 127.9931335
15: -54.2472305, 32.9234695, -54.1260262, 32.9273605, -87.1745911, 87.0494843
16: -68.0361404, 46.2378387, -68.0003128, 45.9905128, -114.0266571, 114.2381439
17: -110.0475006, 43.7993546, -109.7850342, 43.2533569, -153.3008423, 153.5843811
18: -58.5123138, 46.5523186, -58.5136375, 46.4150238, -104.9273376, 105.0659561
19: -46.6289139, 22.7226200, -46.5832825, 22.6362114, -69.2651215, 69.3059006
20: -40.5625534, 32.2649422, -40.4448700, 32.2370262, -72.7995758, 72.7098083
21: -61.3142090, 29.5706081, -61.2469368, 29.4447823, -90.7589874, 90.8175430
22: -67.6796112, 26.5548859, -67.5639496, 26.4910698, -94.1706772, 94.1188278
23: -45.5674438, 33.8423462, -45.5421791, 33.7406807, -79.3081207, 79.3845215
24: -60.1359100, 36.2930794, -60.0218620, 36.2782478, -96.4141541, 96.3149414
25: -47.0103912, 34.0716476, -46.9629707, 34.0008469, -81.0112381, 81.0346146
26: -66.4876099, 52.2130737, -66.3723602, 52.0453568, -118.5329590, 118.5854187
27: -58.7212257, 38.0771408, -58.4652367, 38.0125885, -96.7338104, 96.5423660
28: -43.8821869, 37.2171440, -43.7880745, 37.1987495, -81.0809326, 81.0052185
29: -75.8629913, 29.5502720, -75.7738419, 29.4409275, -105.3039017, 105.3241119
30: -56.2853928, 43.1366272, -56.2417374, 43.0949516, -99.3803406, 99.3783646
31: -61.2926941, 31.6657333, -61.3036079, 31.6027775, -92.8954697, 92.9693298
32: -48.1319733, 38.7754021, -48.0750275, 38.6824188, -86.8143921, 86.8504257
33: -71.6631241, 53.6776199, -71.4935913, 53.6366043, -125.2997131, 125.1712112
34: -58.9648438, 45.2405128, -58.8544312, 45.1841469, -104.1489868, 104.0949249
35: -66.4756317, 46.8792305, -66.2919006, 46.7935982, -113.2692108, 113.1711273
36: -58.8247528, 48.6544724, -58.6513672, 48.5928268, -107.4175797, 107.3058395
37: -75.9918976, 48.8267746, -75.9584732, 48.7609024, -124.7527924, 124.7852478
38: -70.8082504, 57.2681808, -70.6117554, 57.1938438, -128.0020752, 127.8799362
39: -88.1717987, 50.9859848, -88.0008698, 50.9553223, -139.1271210, 138.9868469
40: -58.8038635, 46.6765594, -58.6604805, 46.6932411, -105.4971008, 105.3370361
41: -43.6054993, 42.0896454, -43.5232887, 41.9731598, -85.5786591, 85.6129303
42: -32.8448486, 40.5978165, -32.6745415, 40.3213272, -73.1661758, 73.2723541

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3056709
time: 199.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3063394
time: 109.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -85.1161194, 35.1280212, -84.9380493, 35.0016479, -120.1177673, 120.0660706
1: -48.1646538, 33.0907707, -48.0280457, 32.9885559, -81.1532135, 81.1188202
2: -41.0391922, 28.5817719, -40.8101959, 28.4881897, -69.5273819, 69.3919601
3: -46.4322281, 38.3881073, -46.2012939, 38.2652359, -84.6974564, 84.5894012
4: -53.5151215, 36.7388115, -53.2362938, 36.6332741, -90.1483917, 89.9750977
5: -46.1774826, 40.4525223, -45.9933243, 40.3345604, -86.5120392, 86.4458466
6: -41.9996529, 40.8740082, -41.8506317, 40.6387787, -82.6384277, 82.7246399
7: -54.5265236, 41.0616074, -54.3738556, 40.9754372, -95.5019455, 95.4354553
8: -56.6847229, 48.1085854, -56.3724670, 47.9405899, -104.6253128, 104.4810486
9: -44.0394974, 41.2561798, -43.9420776, 41.0384789, -85.0779724, 85.1982574
10: -65.4300156, 52.7296562, -65.1547928, 52.1803169, -117.6103363, 117.8844147
11: -62.5459061, 40.7288284, -62.3862724, 40.3964348, -102.9423370, 103.1150970
12: -50.5854797, 49.6902618, -50.3096809, 49.1884041, -99.7738724, 99.9999390
13: -64.4628143, 54.9643860, -64.3711929, 54.7321053, -119.1949081, 119.3355713
14: -104.2844391, 23.9959850, -104.0935516, 23.6413651, -127.9257965, 128.0895233
15: -54.3439636, 33.0709915, -54.1430893, 32.9809761, -87.3249359, 87.2140808
16: -68.2441254, 46.3246231, -68.0705261, 46.0015564, -114.2456665, 114.3951492
17: -110.1597366, 43.8776054, -109.8210144, 43.2800827, -153.4398193, 153.6986237
18: -58.7761192, 46.6549911, -58.6013222, 46.4267082, -105.2028275, 105.2563171
19: -46.7960358, 22.7766762, -46.6394539, 22.6433411, -69.4393768, 69.4161301
20: -40.6198959, 32.3169556, -40.4649658, 32.2517242, -72.8716202, 72.7819214
21: -61.4286003, 29.6166916, -61.2853661, 29.4576530, -90.8862457, 90.9020538
22: -67.7681885, 26.6093502, -67.5928650, 26.5151424, -94.2833252, 94.2022171
23: -45.6780090, 33.9066315, -45.5809784, 33.7530327, -79.4310455, 79.4876099
24: -60.2858772, 36.3381996, -60.0716095, 36.2895584, -96.5754318, 96.4098053
25: -47.1304283, 34.1074371, -47.0035362, 34.0116425, -81.1420746, 81.1109772
26: -66.5740814, 52.2736816, -66.4080963, 52.0637894, -118.6378708, 118.6817780
27: -58.8012428, 38.1310883, -58.4906769, 38.0299759, -96.8312225, 96.6217651
28: -43.9636307, 37.2769966, -43.8169022, 37.2138710, -81.1775055, 81.0939026
29: -75.9652863, 29.5941334, -75.8055115, 29.4580746, -105.4233627, 105.3996429
30: -56.3851776, 43.2210045, -56.2755966, 43.1171951, -99.5023727, 99.4965973
31: -61.5733147, 31.7568817, -61.3977432, 31.6128311, -93.1861420, 93.1546173
32: -48.2503853, 38.8448715, -48.1161537, 38.6983719, -86.9487610, 86.9610291
33: -71.7842407, 53.7300758, -71.5349274, 53.6478958, -125.4321365, 125.2649918
34: -59.0998955, 45.3083534, -58.9002724, 45.1965866, -104.2964783, 104.2086258
35: -66.5879288, 46.9337387, -66.3297424, 46.8038902, -113.3918152, 113.2634811
36: -58.8812332, 48.6801453, -58.6723061, 48.6000748, -107.4813080, 107.3524475
37: -76.2887039, 48.9031296, -76.0595703, 48.7700500, -125.0587540, 124.9626999
38: -70.9039612, 57.3301582, -70.6435394, 57.2088966, -128.1128540, 127.9736938
39: -88.2916565, 51.0244217, -88.0403137, 50.9681244, -139.2597809, 139.0647278
40: -59.0277977, 46.7555618, -58.7354813, 46.7031174, -105.7309113, 105.4910431
41: -43.7585678, 42.1576805, -43.5769997, 41.9850464, -85.7436142, 85.7346802
42: -32.9069023, 40.6540375, -32.6975555, 40.3349686, -73.2418671, 73.3515930

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2792311, upper bound: 59.2213401
time: 115.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2792311, upper bound: 59.3235653
time: 93.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -84.9661102, 35.0011101, -85.1605072, 35.1833115, -120.1494217, 120.1616135
1: -48.0323868, 32.9922905, -48.2231445, 33.1357727, -81.1681595, 81.2154388
2: -40.8104935, 28.5221214, -41.1493301, 28.5755386, -69.3860321, 69.6714478
3: -46.2012558, 38.2990189, -46.5440826, 38.4015770, -84.6028290, 84.8431015
4: -53.2405548, 36.6848068, -53.6409149, 36.7220345, -89.9625854, 90.3257141
5: -45.9952164, 40.3649673, -46.2648010, 40.4658241, -86.4610367, 86.6297607
6: -41.8946228, 40.6399994, -42.0181046, 40.9853363, -82.8799591, 82.6581039
7: -54.3789711, 40.9808960, -54.5890732, 41.0953064, -95.4742737, 95.5699615
8: -56.3749962, 48.0100403, -56.8282928, 48.0863876, -104.4613800, 104.8383331
9: -43.9540405, 41.0530319, -44.0580444, 41.3395844, -85.2936249, 85.1110687
10: -65.2202148, 52.1855469, -65.4283752, 52.9849701, -118.2051849, 117.6139221
11: -62.4312820, 40.3937111, -62.5476913, 40.8941269, -103.3254089, 102.9414062
12: -50.4322853, 49.1825867, -50.5090790, 49.9384613, -100.3707428, 99.6916656
13: -64.3735657, 54.7502670, -64.5004272, 55.0459976, -119.4195633, 119.2506866
14: -104.1370544, 23.6386471, -104.3046951, 24.1715927, -128.3086395, 127.9433441
15: -54.1246681, 33.0122375, -54.4542465, 33.0981941, -87.2228622, 87.4664764
16: -68.1060715, 45.9870911, -68.2729340, 46.4996071, -114.6056824, 114.2600250
17: -109.9793091, 43.2732239, -110.0521774, 44.1721649, -154.1514740, 153.3254089
18: -58.6357269, 46.4283066, -58.8052483, 46.7608910, -105.3966141, 105.2335434
19: -46.6822433, 22.6472588, -46.8000107, 22.8341732, -69.5164185, 69.4472656
20: -40.4824905, 32.2549438, -40.6655960, 32.3442192, -72.8266983, 72.9205399
21: -61.3118591, 29.4557610, -61.4588623, 29.6981316, -91.0099945, 90.9146271
22: -67.6006622, 26.5224228, -67.8268433, 26.6688786, -94.2695389, 94.3492661
23: -45.6056366, 33.7530479, -45.7006645, 33.9807892, -79.5864182, 79.4537048
24: -60.0858459, 36.2992020, -60.3666420, 36.3452377, -96.4310684, 96.6658478
25: -47.0213623, 34.0145836, -47.1632805, 34.1435471, -81.1649017, 81.1778641
26: -66.4154816, 52.0647964, -66.6254349, 52.3695221, -118.7849960, 118.6902237
27: -58.5030937, 38.0545158, -58.9271774, 38.1214828, -96.6245728, 96.9816895
28: -43.8314438, 37.2188416, -44.0123825, 37.2960663, -81.1275024, 81.2312164
29: -75.8167267, 29.4593849, -76.0206146, 29.6662903, -105.4830170, 105.4799957
30: -56.2863350, 43.1169968, -56.4217033, 43.2558594, -99.5421906, 99.5386963
31: -61.4385376, 31.6141033, -61.6003380, 31.8241005, -93.2626343, 93.2144394
32: -48.1431351, 38.6934586, -48.2695999, 38.9205475, -87.0636826, 86.9630585
33: -71.5502701, 53.6582565, -71.8812256, 53.7536011, -125.3038712, 125.5394669
34: -58.9074821, 45.2032661, -59.1851692, 45.3415909, -104.2490692, 104.3884277
35: -66.3380280, 46.8291779, -66.6996613, 46.9381866, -113.2762146, 113.5288391
36: -58.6782379, 48.6090546, -58.9717865, 48.7009048, -107.3791275, 107.5808411
37: -76.0894775, 48.7721252, -76.3513107, 48.9601021, -125.0495758, 125.1234360
38: -70.6599808, 57.2204552, -71.0054474, 57.3615074, -128.0214844, 128.2258911
39: -88.0626373, 50.9831123, -88.3580170, 51.0342865, -139.0969238, 139.3411255
40: -58.7560387, 46.7077065, -59.1369591, 46.7773170, -105.5333557, 105.8446655
41: -43.6263046, 41.9832802, -43.7708168, 42.2429657, -85.8692703, 85.7540970
42: -32.7864609, 40.3312759, -32.8532639, 40.8116150, -73.5980759, 73.1845398

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2791744, upper bound: 59.1943546
time: 134.15 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2962831, upper bound: 59.2963424
time: 108.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.1886292, 35.1840210, -85.1605072, 35.1833115, -120.3719406, 120.3445206
1: -48.2275314, 33.1395531, -48.2231445, 33.1357727, -81.3633041, 81.3627014
2: -41.1493759, 28.6091022, -41.1493301, 28.5755386, -69.7249069, 69.7584229
3: -46.5448647, 38.4339485, -46.5440826, 38.4015770, -84.9464340, 84.9780273
4: -53.6442528, 36.7777100, -53.6409149, 36.7220345, -90.3662872, 90.4186249
5: -46.2670746, 40.4958878, -46.2648010, 40.4658241, -86.7328949, 86.7606812
6: -42.0629120, 40.9854736, -42.0181046, 40.9853363, -83.0482483, 83.0035782
7: -54.5959358, 41.1004410, -54.5890732, 41.0953064, -95.6912384, 95.6895142
8: -56.8317719, 48.1577454, -56.8282928, 48.0863876, -104.9181595, 104.9860382
9: -44.0700760, 41.3543663, -44.0580444, 41.3395844, -85.4096603, 85.4124146
10: -65.4940186, 52.9896088, -65.4283752, 52.9849701, -118.4789886, 118.4179840
11: -62.6010017, 40.8908615, -62.5476913, 40.8941269, -103.4951248, 103.4385529
12: -50.6326332, 49.9323273, -50.5090790, 49.9384613, -100.5710831, 100.4414062
13: -64.5028229, 55.0632248, -64.5004272, 55.0459976, -119.5488205, 119.5636292
14: -104.3495331, 24.1683998, -104.3046951, 24.1715927, -128.5211182, 128.4730988
15: -54.4358139, 33.1323128, -54.4542465, 33.0981941, -87.5340042, 87.5865479
16: -68.3146210, 46.4847412, -68.2729340, 46.4996071, -114.8142090, 114.7576675
17: -110.2099304, 44.1660652, -110.0521774, 44.1721649, -154.3820953, 154.2182312
18: -58.8388672, 46.7630920, -58.8052483, 46.7608910, -105.5997391, 105.5683365
19: -46.8426056, 22.8387871, -46.8000107, 22.8341732, -69.6767807, 69.6387939
20: -40.6836128, 32.3471451, -40.6655960, 32.3442192, -73.0278244, 73.0127411
21: -61.4855232, 29.6957874, -61.4588623, 29.6981316, -91.1836548, 91.1546478
22: -67.8338013, 26.6800747, -67.8268433, 26.6688786, -94.5026779, 94.5069199
23: -45.7245865, 33.9806137, -45.7006645, 33.9807892, -79.7053757, 79.6812744
24: -60.3797646, 36.3560753, -60.3666420, 36.3452377, -96.7249985, 96.7227173
25: -47.1807098, 34.1489067, -47.1632805, 34.1435471, -81.3242493, 81.3121872
26: -66.6325836, 52.3704071, -66.6254349, 52.3695221, -119.0020981, 118.9958344
27: -58.9394875, 38.1517220, -58.9271774, 38.1214828, -97.0609741, 97.0788879
28: -44.0269623, 37.3020973, -44.0123825, 37.2960663, -81.3230286, 81.3144836
29: -76.0309906, 29.6689377, -76.0206146, 29.6662903, -105.6972809, 105.6895447
30: -56.4336777, 43.2562256, -56.4217033, 43.2558594, -99.6895370, 99.6779251
31: -61.6419983, 31.8257332, -61.6003380, 31.8241005, -93.4660950, 93.4260712
32: -48.2987289, 38.9148483, -48.2695999, 38.9205475, -87.2192764, 87.1844482
33: -71.8962479, 53.7658730, -71.8812256, 53.7536011, -125.6498489, 125.6470947
34: -59.1923828, 45.3489037, -59.1851692, 45.3415909, -104.5339737, 104.5340729
35: -66.7075348, 46.9635544, -66.6996613, 46.9381866, -113.6457214, 113.6632156
36: -58.9780121, 48.7109222, -58.9717865, 48.7009048, -107.6789169, 107.6827087
37: -76.3798065, 48.9631119, -76.3513107, 48.9601021, -125.3399048, 125.3144226
38: -71.0228271, 57.3729362, -71.0054474, 57.3615074, -128.3843384, 128.3783875
39: -88.3798828, 51.0503654, -88.3580170, 51.0342865, -139.4141693, 139.4083862
40: -59.1587257, 46.7820816, -59.1369591, 46.7773170, -105.9360352, 105.9190369
41: -43.8226814, 42.2408524, -43.7708168, 42.2429657, -86.0656433, 86.0116730
42: -32.9462433, 40.8074799, -32.8532639, 40.8116150, -73.7578583, 73.6607437

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2791744, upper bound: 59.2677182
time: 194.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2962831, upper bound: 59.1942986
time: 442.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 639.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.1757885, upper bound: 59.2854755
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.1757885, upper bound: 59.2854755
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782919, upper bound: 59.2005947
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782919, upper bound: 59.2005947
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782170, upper bound: 59.1729354
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782170, upper bound: 59.1729354
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782170, upper bound: 59.2507654
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2782170, upper bound: 59.3530870
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3056709
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3063394
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2792311, upper bound: 59.2213401
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2792311, upper bound: 59.3235653
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2791744, upper bound: 59.1943546
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2962831, upper bound: 59.2963424
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2791744, upper bound: 59.2677182
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 639.41
Output dim: 6, lower bound: -59.2962831, upper bound: 59.1942986

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -84.8261719, 34.9898529, -84.7795944, 34.8997002, -119.7258606, 119.7694473
1: -47.9857407, 32.8702240, -47.9186783, 32.8708801, -80.8566132, 80.7888947
2: -40.7703896, 28.3488197, -40.5996399, 28.3975105, -69.1678925, 68.9484558
3: -46.1367683, 38.0367355, -45.9834480, 38.1207886, -84.2575531, 84.0201721
4: -53.1787682, 36.4082565, -52.9936371, 36.5098648, -89.6886292, 89.4018936
5: -45.9154205, 40.2006531, -45.7896347, 40.2210541, -86.1364746, 85.9902802
6: -41.6666489, 40.6012192, -41.7061539, 40.4473114, -82.1139526, 82.3073730
7: -54.3193321, 40.8195343, -54.2300072, 40.8640671, -95.1833954, 95.0495453
8: -56.3256531, 47.6329231, -56.1190491, 47.7655716, -104.0912170, 103.7519684
9: -43.9338837, 41.0429993, -43.8813095, 40.8777466, -84.8116302, 84.9243088
10: -65.1591187, 52.3086739, -65.0351562, 51.8264008, -116.9855118, 117.3438263
11: -62.2848320, 40.4246063, -62.2715149, 40.1432266, -102.4280548, 102.6961136
12: -50.2023697, 49.1687546, -50.1809502, 48.7425919, -98.9449615, 99.3496857
13: -64.2952576, 54.5782661, -64.2837677, 54.5100861, -118.8053284, 118.8620300
14: -104.0151978, 23.6796207, -103.9519348, 23.3541756, -127.3693695, 127.6315536
15: -54.1328430, 32.8221588, -53.9920654, 32.8607864, -86.9936295, 86.8142242
16: -67.8967285, 46.0658951, -67.9015961, 45.8172073, -113.7139282, 113.9674911
17: -109.8016510, 43.3565788, -109.6972351, 42.8083458, -152.6099854, 153.0538177
18: -58.4003220, 46.3898849, -58.4366951, 46.2564087, -104.6567307, 104.8265839
19: -46.5241928, 22.6164188, -46.5212708, 22.5356216, -69.0598145, 69.1376801
20: -40.4433823, 32.2080345, -40.3462296, 32.1830559, -72.6264343, 72.5542603
21: -61.2132149, 29.4341316, -61.1733856, 29.3065605, -90.5197601, 90.6075134
22: -67.5813522, 26.4356976, -67.4726410, 26.3822632, -93.9636002, 93.9083328
23: -45.4826736, 33.7188606, -45.4815445, 33.6201515, -79.1028290, 79.2004089
24: -59.9939690, 36.2546196, -59.8945274, 36.2513123, -96.2452850, 96.1491470
25: -46.9275665, 33.9937439, -46.8982086, 33.9294662, -80.8570328, 80.8919525
26: -66.3880692, 52.0270233, -66.2811279, 51.8621483, -118.2502136, 118.3081512
27: -58.5371895, 38.0177307, -58.2961197, 37.9838028, -96.5209961, 96.3138504
28: -43.7970505, 37.1669731, -43.7182350, 37.1545792, -80.9516296, 80.8852081
29: -75.7708054, 29.3975430, -75.6929474, 29.2921658, -105.0629730, 105.0904922
30: -56.2033043, 43.0576172, -56.1721916, 43.0179482, -99.2212448, 99.2298126
31: -61.1624413, 31.5742493, -61.2149239, 31.5139008, -92.6763458, 92.7891693
32: -48.0452271, 38.6283188, -48.0181808, 38.5303345, -86.5755539, 86.6464996
33: -71.5015411, 53.6099739, -71.3483429, 53.5817947, -125.0833282, 124.9583130
34: -58.8429718, 45.1625023, -58.7412415, 45.1152573, -103.9582291, 103.9037399
35: -66.3330536, 46.8154030, -66.1581573, 46.7551651, -113.0882187, 112.9735565
36: -58.7324867, 48.5876350, -58.5662041, 48.5348282, -107.2673187, 107.1538391
37: -75.8499908, 48.6469727, -75.8461685, 48.5853500, -124.4353409, 124.4931412
38: -70.6610260, 57.1362801, -70.4843063, 57.0760422, -127.7370682, 127.6205826
39: -88.0190277, 50.9263382, -87.8711243, 50.9119987, -138.9310303, 138.7974548
40: -58.6225395, 46.6220360, -58.5037804, 46.6440697, -105.2666092, 105.1258087
41: -43.4781494, 41.9286423, -43.4487877, 41.8107834, -85.2889328, 85.3774261
42: -32.7020111, 40.3635788, -32.6251183, 40.0841217, -72.7861328, 72.9886932

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2710483
time: 156.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2760028
time: 187.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -84.8261719, 34.9898529, -84.9215012, 34.9778786, -119.8040466, 119.9113464
1: -47.9857407, 32.8702240, -48.0198746, 32.9401855, -80.9259186, 80.8900909
2: -40.7703896, 28.3488197, -40.7991219, 28.4648056, -69.2351837, 69.1479416
3: -46.1367683, 38.0367355, -46.1922607, 38.2089500, -84.3457031, 84.2289963
4: -53.1787682, 36.4082565, -53.2281952, 36.6049690, -89.7837372, 89.6364517
5: -45.9154205, 40.2006531, -45.9841766, 40.3075142, -86.2229309, 86.1848297
6: -41.6666489, 40.6012192, -41.8184204, 40.6247711, -82.2914200, 82.4196320
7: -54.3193321, 40.8195343, -54.3680687, 40.9131813, -95.2325134, 95.1876068
8: -56.3256531, 47.6329231, -56.3636131, 47.8898392, -104.2154922, 103.9965363
9: -43.9338837, 41.0429993, -43.9385376, 41.0343437, -84.9682312, 84.9815369
10: -65.1591187, 52.3086739, -65.1788101, 52.1701126, -117.3292313, 117.4874802
11: -62.2848320, 40.4246063, -62.3830528, 40.3781395, -102.6629715, 102.8076477
12: -50.2023697, 49.1687546, -50.3609886, 49.1648865, -99.3672562, 99.5297241
13: -64.2952576, 54.5782661, -64.3540955, 54.6802559, -118.9755096, 118.9323578
14: -104.0151978, 23.6796207, -104.0971527, 23.6124191, -127.6276093, 127.7767715
15: -54.1328430, 32.8221588, -54.1075516, 32.9583740, -87.0912018, 86.9297104
16: -67.8967285, 46.0658951, -68.0355988, 45.9759941, -113.8727188, 114.1014938
17: -109.8016510, 43.3565788, -109.9432297, 43.2464752, -153.0481262, 153.2998047
18: -58.4003220, 46.3898849, -58.5478973, 46.4166107, -104.8169250, 104.9377747
19: -46.5241928, 22.6164188, -46.6259460, 22.6401196, -69.1643143, 69.2423630
20: -40.4433823, 32.2080345, -40.4623375, 32.2402344, -72.6836090, 72.6703720
21: -61.2132149, 29.4341316, -61.2733231, 29.4428520, -90.6560669, 90.7074585
22: -67.5813522, 26.4356976, -67.5716705, 26.4980927, -94.0794373, 94.0073700
23: -45.4826736, 33.7188606, -45.5666924, 33.7406693, -79.2233353, 79.2855530
24: -59.9939690, 36.2546196, -60.0359840, 36.2878342, -96.2817993, 96.2906036
25: -46.9275665, 33.9937439, -46.9806976, 34.0037460, -80.9313126, 80.9744415
26: -66.3880692, 52.0270233, -66.3796844, 52.0462761, -118.4343414, 118.4067078
27: -58.5371895, 38.0177307, -58.4775925, 38.0370827, -96.5742722, 96.4953232
28: -43.7970505, 37.1669731, -43.8025284, 37.2036781, -81.0007324, 80.9694977
29: -75.7708054, 29.3975430, -75.7849731, 29.4420700, -105.2128754, 105.1825180
30: -56.2033043, 43.0576172, -56.2524261, 43.0947151, -99.2980042, 99.3100433
31: -61.1624413, 31.5742493, -61.3442345, 31.6040268, -92.7664642, 92.9184875
32: -48.0452271, 38.6283188, -48.1019173, 38.6774521, -86.7226791, 86.7302399
33: -71.5015411, 53.6099739, -71.5088272, 53.6469231, -125.1484604, 125.1187897
34: -58.8429718, 45.1625023, -58.8615417, 45.1908035, -104.0337753, 104.0240479
35: -66.3330536, 46.8154030, -66.3000946, 46.8188248, -113.1518784, 113.1154861
36: -58.7324867, 48.5876350, -58.6572266, 48.6017761, -107.3342590, 107.2448578
37: -75.8499908, 48.6469727, -75.9881363, 48.7629547, -124.6129379, 124.6351089
38: -70.6610260, 57.1362801, -70.6281128, 57.2053757, -127.8663940, 127.7643814
39: -88.0190277, 50.9263382, -88.0231781, 50.9702644, -138.9892883, 138.9495239
40: -58.6225395, 46.6220360, -58.6808701, 46.6977997, -105.3203430, 105.3029022
41: -43.4781494, 41.9286423, -43.5724564, 41.9713821, -85.4495316, 85.5010986
42: -32.7020111, 40.3635788, -32.7633400, 40.3175888, -73.0195999, 73.1269226

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2710483
time: 133.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2760028
time: 116.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.9739532, 35.0478477, -84.7502975, 34.9203796, -119.8943329, 119.7981415
1: -48.0622482, 33.0211487, -47.9200821, 32.8147888, -80.8770294, 80.9412308
2: -40.8387642, 28.5146618, -40.6779137, 28.3078194, -69.1465759, 69.1925735
3: -46.2202454, 38.3006096, -46.0510101, 37.9778595, -84.1981049, 84.3516235
4: -53.2765121, 36.6383591, -53.0641327, 36.3834114, -89.6599197, 89.7024918
5: -45.9806595, 40.3661194, -45.8656082, 40.1470108, -86.1276703, 86.2317276
6: -41.8855362, 40.6946487, -41.6035461, 40.4886017, -82.3741379, 82.2981949
7: -54.3850784, 41.0125771, -54.2659798, 40.7645950, -95.1496582, 95.2785568
8: -56.4374237, 47.9828072, -56.1842880, 47.5645676, -104.0019760, 104.1670761
9: -43.9817848, 41.0978088, -43.8784485, 40.9377289, -84.9195099, 84.9762573
10: -65.2857513, 52.3846893, -64.9973526, 51.9964867, -117.2822189, 117.3820419
11: -62.4255905, 40.4930801, -62.2185974, 40.2522011, -102.6777954, 102.7116776
12: -50.4036179, 49.2664413, -50.0842552, 48.9550781, -99.3586960, 99.3506927
13: -64.3912048, 54.7892380, -64.2534027, 54.4728622, -118.8640594, 119.0426331
14: -104.1371918, 23.7361469, -103.9330444, 23.5000496, -127.6372299, 127.6691895
15: -54.2295532, 32.9696960, -54.0124359, 32.8014679, -87.0310211, 86.9821320
16: -68.1047134, 46.1526794, -67.8238525, 45.8649559, -113.9696655, 113.9765320
17: -109.9136963, 43.4349518, -109.6781540, 43.0609360, -152.9746094, 153.1130981
18: -58.6641541, 46.4925537, -58.3076019, 46.2728806, -104.9370270, 104.8001556
19: -46.6912994, 22.6704979, -46.4500198, 22.5568886, -69.2481842, 69.1205139
20: -40.5007401, 32.2600288, -40.3751602, 32.1820297, -72.6827621, 72.6351852
21: -61.3275757, 29.4802151, -61.1434593, 29.3670120, -90.6945877, 90.6236725
22: -67.6698608, 26.4901657, -67.4769592, 26.4158382, -94.0857010, 93.9671173
23: -45.5932770, 33.7831573, -45.4440842, 33.6497307, -79.2430038, 79.2272415
24: -60.1438141, 36.2997055, -59.8777122, 36.2344627, -96.3782806, 96.1774063
25: -47.0476418, 34.0295639, -46.8598137, 33.9526329, -81.0002747, 80.8893738
26: -66.4745026, 52.0876808, -66.2953262, 51.9445190, -118.4190216, 118.3830109
27: -58.6172447, 38.0716171, -58.3567276, 37.9658585, -96.5830994, 96.4283295
28: -43.8785553, 37.2268600, -43.7104721, 37.1393051, -81.0178604, 80.9373322
29: -75.8731079, 29.4414673, -75.6773605, 29.3607960, -105.2339020, 105.1188202
30: -56.3030777, 43.1420059, -56.1529045, 43.0110893, -99.3141479, 99.2949066
31: -61.4432297, 31.6654129, -61.0817566, 31.4919319, -92.9351501, 92.7471695
32: -48.1635246, 38.6977654, -47.9765472, 38.5816383, -86.7451630, 86.6743088
33: -71.6227417, 53.6623611, -71.3648071, 53.5787964, -125.2015381, 125.0271683
34: -58.9780922, 45.2303314, -58.7264519, 45.1084328, -104.0865173, 103.9567719
35: -66.4453735, 46.8697968, -66.1732559, 46.7372055, -113.1825790, 113.0430527
36: -58.7889748, 48.6132889, -58.5896149, 48.5557175, -107.3446960, 107.2029037
37: -76.1468353, 48.7233658, -75.7203064, 48.6375313, -124.7843628, 124.4436722
38: -70.7567596, 57.1982841, -70.5060501, 57.1090279, -127.8657837, 127.7043304
39: -88.1389389, 50.9647827, -87.8780212, 50.9156036, -139.0545349, 138.8428040
40: -58.8466263, 46.7010117, -58.4565315, 46.6079102, -105.4545364, 105.1575470
41: -43.6312027, 41.9966965, -43.3956604, 41.8654900, -85.4966812, 85.3923569
42: -32.7640305, 40.4197769, -32.6152840, 40.2035751, -72.9676056, 73.0350647

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1839258, upper bound: 59.1861574
time: 141.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2687865, upper bound: 59.1911254
time: 130.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.9739532, 35.0478477, -84.8981323, 34.9783936, -119.9523315, 119.9459839
1: -48.0622482, 33.0211487, -47.9965744, 32.9657860, -81.0280304, 81.0177231
2: -40.8387642, 28.5146618, -40.7463074, 28.4736481, -69.3124084, 69.2609711
3: -46.2202454, 38.3006096, -46.1345329, 38.2417831, -84.4620209, 84.4351425
4: -53.2765121, 36.6383591, -53.1618729, 36.6135178, -89.8900299, 89.8002319
5: -45.9806595, 40.3661194, -45.9308777, 40.3125076, -86.2931671, 86.2969894
6: -41.8855362, 40.6946487, -41.8224945, 40.5820732, -82.4676056, 82.5171432
7: -54.3850784, 41.0125771, -54.3316994, 40.9576797, -95.3427582, 95.3442688
8: -56.4374237, 47.9828072, -56.2960663, 47.9144363, -104.3518372, 104.2788696
9: -43.9817848, 41.0978088, -43.9262924, 40.9925346, -84.9743195, 85.0240936
10: -65.2857513, 52.3846893, -65.1239471, 52.0725212, -117.3582764, 117.5086212
11: -62.4255905, 40.4930801, -62.3592796, 40.3206940, -102.7462845, 102.8523560
12: -50.4036179, 49.2664413, -50.2855377, 49.0527916, -99.4564056, 99.5519714
13: -64.3912048, 54.7892380, -64.3493500, 54.6839447, -119.0751343, 119.1385880
14: -104.1371918, 23.7361469, -104.0549393, 23.5566063, -127.6937714, 127.7910843
15: -54.2295532, 32.9696960, -54.1091881, 32.9490356, -87.1785889, 87.0788879
16: -68.1047134, 46.1526794, -68.0317917, 45.9517860, -114.0564880, 114.1844635
17: -109.9136963, 43.4349518, -109.7901459, 43.1393433, -153.0530243, 153.2250977
18: -58.6641541, 46.4925537, -58.5715408, 46.3755493, -105.0396957, 105.0640869
19: -46.6912994, 22.6704979, -46.6171188, 22.6109924, -69.3022919, 69.2876129
20: -40.5007401, 32.2600288, -40.4325829, 32.2339973, -72.7347260, 72.6926117
21: -61.3275757, 29.4802151, -61.2578583, 29.4130898, -90.7406540, 90.7380676
22: -67.6698608, 26.4901657, -67.5655060, 26.4703331, -94.1401978, 94.0556641
23: -45.5932770, 33.7831573, -45.5547333, 33.7140388, -79.3073120, 79.3378830
24: -60.1438141, 36.2997055, -60.0275726, 36.2795105, -96.4233246, 96.3272705
25: -47.0476418, 34.0295639, -46.9799500, 33.9884491, -81.0360870, 81.0095139
26: -66.4745026, 52.0876808, -66.3817825, 52.0052261, -118.4797287, 118.4694672
27: -58.6172447, 38.0716171, -58.4368210, 38.0197182, -96.6369629, 96.5084381
28: -43.8785553, 37.2268600, -43.7920074, 37.1990585, -81.0776138, 81.0188675
29: -75.8731079, 29.4414673, -75.7796631, 29.4047241, -105.2778244, 105.2211304
30: -56.3030777, 43.1420059, -56.2527428, 43.0954285, -99.3984985, 99.3947449
31: -61.4432297, 31.6654129, -61.3627739, 31.5831070, -93.0263367, 93.0281830
32: -48.1635246, 38.6977654, -48.0948257, 38.6510468, -86.8145752, 86.7925873
33: -71.6227417, 53.6623611, -71.4860764, 53.6312027, -125.2539215, 125.1484222
34: -58.9780922, 45.2303314, -58.8616600, 45.1761971, -104.1542816, 104.0919876
35: -66.4453735, 46.8697968, -66.2856140, 46.7915230, -113.2368927, 113.1554108
36: -58.7889748, 48.6132889, -58.6461296, 48.5813293, -107.3703003, 107.2594147
37: -76.1468353, 48.7233658, -76.0172424, 48.7138901, -124.8607254, 124.7406082
38: -70.7567596, 57.1982841, -70.6017914, 57.1709137, -127.9276733, 127.8000717
39: -88.1389389, 50.9647827, -87.9979706, 50.9540520, -139.0929871, 138.9627533
40: -58.8466263, 46.7010117, -58.6807709, 46.6868896, -105.5335159, 105.3817825
41: -43.6312027, 41.9966965, -43.5487366, 41.9335632, -85.5647583, 85.5454254
42: -32.7640305, 40.4197769, -32.6772461, 40.2597504, -73.0237808, 73.0970230

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1839258, upper bound: 59.1861574
time: 202.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1839258, upper bound: 59.2232594
time: 117.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -84.7795944, 34.8997002, -84.9730988, 35.1014633, -119.8810577, 119.8727875
1: -47.9186783, 32.8708801, -48.1144447, 32.9620895, -80.8807678, 80.9853210
2: -40.5996399, 28.3975105, -41.0166550, 28.3951378, -68.9947815, 69.4141541
3: -45.9834480, 38.1207886, -46.3927765, 38.1142998, -84.0977478, 84.5135651
4: -52.9936371, 36.5098648, -53.4679260, 36.4717865, -89.4654160, 89.9777908
5: -45.7896347, 40.2210541, -46.1364517, 40.2782288, -86.0678635, 86.3574982
6: -41.7061539, 40.4473114, -41.7704430, 40.8344193, -82.5405731, 82.2177429
7: -54.2300072, 40.8640671, -54.4798279, 40.8845215, -95.1145172, 95.3438873
8: -56.1190491, 47.7655716, -56.6393280, 47.7103882, -103.8294373, 104.4048996
9: -43.8813095, 40.8777466, -43.9941750, 41.2380447, -85.1193542, 84.8719177
10: -65.0351562, 51.8264008, -65.2709351, 52.8005753, -117.8357315, 117.0973358
11: -62.2715149, 40.1432266, -62.3789406, 40.7494583, -103.0209732, 102.5221710
12: -50.1809502, 48.7425919, -50.2831993, 49.7046204, -99.8855591, 99.0257874
13: -64.2837677, 54.5100861, -64.3820114, 54.7834816, -119.0672455, 118.8920898
14: -103.9519348, 23.3541756, -104.1439667, 24.0293198, -127.9812393, 127.4981384
15: -53.9920654, 32.8607864, -54.3251419, 32.9180222, -86.9100800, 87.1859283
16: -67.9015961, 45.8172073, -68.0263214, 46.3580513, -114.2596436, 113.8435287
17: -109.6972351, 42.8083458, -109.9089737, 43.9514542, -153.6486816, 152.7173157
18: -58.4366951, 46.2564087, -58.5116806, 46.6062279, -105.0429230, 104.7680817
19: -46.5212708, 22.5356216, -46.6106186, 22.7472439, -69.2685089, 69.1462402
20: -40.3462296, 32.1830559, -40.5745773, 32.2745552, -72.6207886, 72.7576294
21: -61.1733856, 29.3065605, -61.3164330, 29.6073380, -90.7807236, 90.6229935
22: -67.4726410, 26.3822632, -67.7117767, 26.5696239, -94.0422668, 94.0940399
23: -45.4815445, 33.6201515, -45.5637054, 33.8763046, -79.3578491, 79.1838531
24: -59.8945274, 36.2513123, -60.1722260, 36.2897339, -96.1842575, 96.4235382
25: -46.8982086, 33.9294662, -47.0194397, 34.0835037, -80.9817123, 80.9488983
26: -66.2811279, 51.8621483, -66.5122375, 52.2498627, -118.5309906, 118.3743744
27: -58.2961197, 37.9838028, -58.7921829, 38.0571594, -96.3532715, 96.7759857
28: -43.7182350, 37.1545792, -43.9057159, 37.2212830, -80.9395142, 81.0602951
29: -75.6929474, 29.2921658, -75.8920746, 29.5676651, -105.2606125, 105.1842346
30: -56.1721916, 43.0179482, -56.2984962, 43.1494446, -99.3216400, 99.3164444
31: -61.2149239, 31.5139008, -61.2846489, 31.7026806, -92.9175949, 92.7985535
32: -48.0181808, 38.5303345, -48.1293564, 38.8037224, -86.8218994, 86.6596909
33: -71.3483429, 53.5817947, -71.7108612, 53.6841660, -125.0325089, 125.2926407
34: -58.7412415, 45.1152573, -59.0109444, 45.2529068, -103.9941483, 104.1262054
35: -66.1581573, 46.7551651, -66.5429382, 46.8712120, -113.0293579, 113.2980957
36: -58.5662041, 48.5348282, -58.8889160, 48.6568680, -107.2230682, 107.4237442
37: -75.8461685, 48.5853500, -76.0120926, 48.8268509, -124.6730194, 124.5974426
38: -70.4843063, 57.0760422, -70.8672409, 57.2595062, -127.7438126, 127.9432755
39: -87.8711243, 50.9119987, -88.1952438, 50.9816704, -138.8527832, 139.1072388
40: -58.5037804, 46.6440697, -58.8568192, 46.6817474, -105.1855316, 105.5008850
41: -43.4487877, 41.8107834, -43.5890961, 42.1232376, -85.5720215, 85.3998718
42: -32.6251183, 40.0841217, -32.7704391, 40.6796913, -73.3048096, 72.8545609

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2961248, upper bound: 59.1584117
time: 192.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2687151, upper bound: 59.1634016
time: 622.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -84.8240509, 34.9228973, -85.1209106, 35.1594315, -119.9834747, 120.0438080
1: -47.9311523, 32.9229126, -48.1909790, 33.1128654, -81.0440216, 81.1138916
2: -40.6109810, 28.4547215, -41.0850220, 28.5609379, -69.1719208, 69.5397415
3: -45.9924240, 38.2106895, -46.4762535, 38.3780937, -84.3705139, 84.6869431
4: -53.0059662, 36.5895386, -53.5657425, 36.7018509, -89.7078094, 90.1552811
5: -45.8006439, 40.2783737, -46.2016830, 40.4436378, -86.2442780, 86.4800568
6: -41.7821960, 40.4625092, -41.9892769, 40.9278564, -82.7100525, 82.4517822
7: -54.2408447, 40.9316864, -54.5455513, 41.0775299, -95.3183746, 95.4772339
8: -56.1304016, 47.8855286, -56.7510300, 48.0602455, -104.1906433, 104.6365509
9: -43.8967133, 40.8963776, -44.0421944, 41.2928581, -85.1895599, 84.9385681
10: -65.0763779, 51.8417664, -65.3974609, 52.8766022, -117.9529800, 117.2392120
11: -62.3195457, 40.1588058, -62.5197906, 40.8180084, -103.1375504, 102.6785965
12: -50.2520790, 48.7602615, -50.4844284, 49.8022614, -100.0543365, 99.2446899
13: -64.3031693, 54.5800552, -64.4779968, 54.9942551, -119.2974243, 119.0580521
14: -103.9915619, 23.3803215, -104.2660599, 24.0861130, -128.0776672, 127.6463776
15: -54.0091057, 32.9143906, -54.4218826, 33.0655518, -87.0746536, 87.3362732
16: -67.9718018, 45.8282585, -68.2342377, 46.4448471, -114.4166489, 114.0625000
17: -109.7331390, 42.8350754, -110.0212097, 44.0297089, -153.7628479, 152.8562927
18: -58.5244064, 46.2680969, -58.7754440, 46.7089233, -105.2333221, 105.0435333
19: -46.5774460, 22.5427513, -46.7776794, 22.8013554, -69.3787994, 69.3204346
20: -40.3663330, 32.1977310, -40.6318970, 32.3265953, -72.6929321, 72.8296280
21: -61.2117920, 29.3194389, -61.4308548, 29.6534233, -90.8652191, 90.7502899
22: -67.5015411, 26.4063301, -67.8003998, 26.6240826, -94.1256104, 94.2067261
23: -45.5203400, 33.6325035, -45.6742592, 33.9406357, -79.4609756, 79.3067627
24: -59.9442940, 36.2625961, -60.3221474, 36.3348846, -96.2791748, 96.5847397
25: -46.9388046, 33.9402733, -47.1394424, 34.1193008, -81.0581055, 81.0797119
26: -66.3167877, 51.8806305, -66.5987701, 52.3105392, -118.6273193, 118.4794006
27: -58.3216019, 38.0011597, -58.8722687, 38.1111107, -96.4327087, 96.8734131
28: -43.7470627, 37.1697311, -43.9871521, 37.2812195, -81.0282822, 81.1568832
29: -75.7246094, 29.3093243, -75.9944763, 29.6115551, -105.3361664, 105.3038025
30: -56.2059975, 43.0401878, -56.3983116, 43.2338295, -99.4398193, 99.4384918
31: -61.3090515, 31.5239601, -61.5652084, 31.7938576, -93.1029053, 93.0891724
32: -48.0592995, 38.5462875, -48.2476501, 38.8731766, -86.9324799, 86.7939377
33: -71.3896942, 53.5930710, -71.8318710, 53.7366447, -125.1263428, 125.4249344
34: -58.7871399, 45.1276703, -59.1459999, 45.3208313, -104.1079712, 104.2736511
35: -66.1960449, 46.7654495, -66.6552277, 46.9257431, -113.1217804, 113.4206772
36: -58.5871544, 48.5420532, -58.9453659, 48.6825790, -107.2697144, 107.4874191
37: -75.9472809, 48.5944901, -76.3087921, 48.9031982, -124.8504791, 124.9032745
38: -70.5160904, 57.0910873, -70.9629059, 57.3215828, -127.8376770, 128.0539856
39: -87.9105453, 50.9248123, -88.3151398, 51.0201035, -138.9306335, 139.2399445
40: -58.5788536, 46.6539268, -59.0806541, 46.7607574, -105.3396072, 105.7345810
41: -43.5024948, 41.8226624, -43.7421036, 42.1912994, -85.6937943, 85.5647659
42: -32.6481171, 40.0977631, -32.8325272, 40.7360229, -73.3841400, 72.9302826

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1933611, upper bound: 59.2577647
time: 109.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.1933611, upper bound: 59.2747846
time: 119.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -85.0025940, 35.0797005, -84.9730988, 35.1014633, -120.1040497, 120.0527954
1: -48.1119995, 33.0177002, -48.1144447, 32.9620895, -81.0740891, 81.1321411
2: -40.9372330, 28.4848270, -41.0166550, 28.3951378, -69.3323669, 69.5014648
3: -46.3232269, 38.2570724, -46.3927765, 38.1142998, -84.4375305, 84.6498489
4: -53.3923111, 36.5976639, -53.4679260, 36.4717865, -89.8640900, 90.0655899
5: -46.0588913, 40.3522034, -46.1364517, 40.2782288, -86.3371201, 86.4886475
6: -41.8722305, 40.7903976, -41.7704430, 40.8344193, -82.7066498, 82.5608368
7: -54.4422684, 40.9838409, -54.4798279, 40.8845215, -95.3267899, 95.4636688
8: -56.5723076, 47.9118958, -56.6393280, 47.7103882, -104.2826996, 104.5512238
9: -43.9970779, 41.1764259, -43.9941750, 41.2380447, -85.2351227, 85.1706009
10: -65.3084412, 52.6289749, -65.2709351, 52.8005753, -118.1090164, 117.8999100
11: -62.4314194, 40.6394539, -62.3789406, 40.7494583, -103.1808777, 103.0183868
12: -50.3792534, 49.4904823, -50.2831993, 49.7046204, -100.0838623, 99.7736816
13: -64.4112244, 54.8160782, -64.3820114, 54.7834816, -119.1947021, 119.1980896
14: -104.1626358, 23.8819637, -104.1439667, 24.0293198, -128.1919556, 128.0259247
15: -54.3035851, 32.9764709, -54.3251419, 32.9180222, -87.2216034, 87.3016129
16: -68.1051025, 46.3012543, -68.0263214, 46.3580513, -114.4631500, 114.3275757
17: -109.9278488, 43.6953888, -109.9089737, 43.9514542, -153.8793030, 153.6043701
18: -58.6394424, 46.5882111, -58.5116806, 46.6062279, -105.2456665, 105.0998917
19: -46.6820908, 22.7251377, -46.6106186, 22.7472439, -69.4293365, 69.3357544
20: -40.5431442, 32.2755928, -40.5745773, 32.2745552, -72.8177032, 72.8501740
21: -61.3457184, 29.5464058, -61.3164330, 29.6073380, -90.9530487, 90.8628387
22: -67.7080078, 26.5367146, -67.7117767, 26.5696239, -94.2776337, 94.2484894
23: -45.6009865, 33.8432770, -45.5637054, 33.8763046, -79.4772949, 79.4069824
24: -60.1877365, 36.3061523, -60.1722260, 36.2897339, -96.4774704, 96.4783783
25: -47.0569916, 34.0584106, -47.0194397, 34.0835037, -81.1404877, 81.0778503
26: -66.4971771, 52.1657944, -66.5122375, 52.2498627, -118.7470322, 118.6780243
27: -58.7292786, 38.0748367, -58.7921829, 38.0571594, -96.7864227, 96.8670197
28: -43.9128571, 37.2363396, -43.9057159, 37.2212830, -81.1341400, 81.1420593
29: -75.9071808, 29.4982338, -75.8920746, 29.5676651, -105.4748459, 105.3903046
30: -56.3168983, 43.1543198, -56.2984962, 43.1494446, -99.4663391, 99.4528122
31: -61.4175911, 31.7237625, -61.2846489, 31.7026806, -93.1202621, 93.0084076
32: -48.1703072, 38.7518539, -48.1293564, 38.8037224, -86.9740295, 86.8812103
33: -71.6932602, 53.6863365, -71.7108612, 53.6841660, -125.3774109, 125.3971863
34: -59.0244217, 45.2581749, -59.0109444, 45.2529068, -104.2773285, 104.2691193
35: -66.5269852, 46.8895416, -66.5429382, 46.8712120, -113.3981934, 113.4324799
36: -58.8647690, 48.6368332, -58.8889160, 48.6568680, -107.5216293, 107.5257416
37: -76.1372833, 48.7730026, -76.0120926, 48.8268509, -124.9641342, 124.7850952
38: -70.8432770, 57.2243233, -70.8672409, 57.2595062, -128.1027527, 128.0915680
39: -88.1875534, 50.9776001, -88.1952438, 50.9816704, -139.1692200, 139.1728516
40: -58.9012260, 46.7171860, -58.8568192, 46.6817474, -105.5829773, 105.5740051
41: -43.6416054, 42.0678635, -43.5890961, 42.1232376, -85.7648468, 85.6569595
42: -32.7800369, 40.5592957, -32.7704391, 40.6796913, -73.4597244, 73.3297272

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1682

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3294568, upper bound: 59.2361534
time: 534.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2687151, upper bound: 59.2407912
time: 110.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 647.74 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2710483
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2760028
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2710483
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1663411, upper bound: 59.2760028
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1839258, upper bound: 59.1861574
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.2687865, upper bound: 59.1911254
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1839258, upper bound: 59.1861574
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1839258, upper bound: 59.2232594
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.2961248, upper bound: 59.1584117
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.2687151, upper bound: 59.1634016
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1933611, upper bound: 59.2577647
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.1933611, upper bound: 59.2747846
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.3294568, upper bound: 59.2361534
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 647.74
Output dim: 6, lower bound: -59.2687151, upper bound: 59.2407912
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2782170, upper bound: 59.3530870
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3056709
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.1729354, upper bound: 59.3063394
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2792311, upper bound: 59.2213401
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2792311, upper bound: 59.3235653
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2791744, upper bound: 59.1943546
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2962831, upper bound: 59.2963424
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2791744, upper bound: 59.2677182
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 647.74
Output dim: 6, lower bound: -59.2962831, upper bound: 59.1942986

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 124.42 + 7193.84 = 7318.27 seconds
