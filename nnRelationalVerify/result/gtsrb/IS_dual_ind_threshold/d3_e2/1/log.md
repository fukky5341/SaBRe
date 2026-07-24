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
execution time: IAR + RelationalAnalysis = 3.03 + 122.02 = 125.05 seconds
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3689607
time: 117.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3855864
time: 160.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 277.99 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 277.99
Output dim: 6, lower bound: -59.3855864, upper bound: 59.3689607
IS_A2, status: Status.UNKNOWN, split count: 1, time: 277.99
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3059511
time: 138.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3562081
time: 126.39 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3267093
time: 131.14 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3734268
time: 118.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 252.31 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 252.31
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3059511
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 252.31
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3562081
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 252.31
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3267093
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 252.31
Output dim: 6, lower bound: -59.2985562, upper bound: 59.3734268

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=196, inp2_unstable=196, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3267094, upper bound: 59.2994946
time: 109.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2994946
time: 107.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 220.20 seconds
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 220.20
Output dim: 6, lower bound: -59.3267094, upper bound: 59.2994946
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 220.20
Output dim: 6, lower bound: -59.3260866, upper bound: 59.2994946

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 125.05 + 1017.64 = 1142.69 seconds
