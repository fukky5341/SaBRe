## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 7200 seconds
Split limit: 100
Threshold: 69.6277649019


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308)
1: (-56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296)
2: (-48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927)
3: (-53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266)
4: (-57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419)
5: (-52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908)
6: (-70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624)
7: (-64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759)
8: (-64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671)
9: (-59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303)
10: (-83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396)
11: (-78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407)
12: (-77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902)
13: (-80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275)
14: (-122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064)
15: (-66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709)
16: (-87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249)
17: (-123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349)
18: (-70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777)
19: (-55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344)
20: (-50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348)
21: (-71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061)
22: (-78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562)
23: (-56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506)
24: (-70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976)
25: (-58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133)
26: (-77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457)
27: (-74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279)
28: (-55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530)
29: (-89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895)
30: (-67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361)
31: (-72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855)
32: (-75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588)
33: (-98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932)
34: (-81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603)
35: (-84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807)
36: (-82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416)
37: (-114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015)
38: (-100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432)
39: (-117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675)
40: (-97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405)
41: (-72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691)
42: (-54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.98 + 119.29 = 122.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -69.6625962, upper bound: 69.6625962

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1241
type: B, layer: 1, pos: 1241
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 1245
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1271
type: B, layer: 1, pos: 1271
type: A, layer: 1, pos: 1255
type: B, layer: 1, pos: 1255
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1246
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1229
type: B, layer: 1, pos: 1229
type: A, layer: 1, pos: 1257
type: B, layer: 1, pos: 1257
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1256
type: B, layer: 1, pos: 1256
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1213
type: B, layer: 1, pos: 1213
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1214
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1233
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1155
type: B, layer: 1, pos: 1155
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1272
type: B, layer: 1, pos: 1272
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1234
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1242
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1277
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1154
type: B, layer: 1, pos: 1154
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1217
type: A, layer: 1, pos: 1217
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1267
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1258
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1251
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1261
type: B, layer: 1, pos: 1261
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1073
type: B, layer: 1, pos: 1073
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1254
type: B, layer: 1, pos: 1254
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1238
type: B, layer: 1, pos: 1238
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1268
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1262
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1252
type: B, layer: 1, pos: 1252
type: A, layer: 1, pos: 1270
type: B, layer: 1, pos: 1270
type: A, layer: 1, pos: 1089
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1236
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1278
type: B, layer: 1, pos: 1278
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1274
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1074
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 1244
type: B, layer: 1, pos: 1244
type: A, layer: 1, pos: 1276
type: B, layer: 1, pos: 1276
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1260
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1228
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1220
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1075
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1058
type: A, layer: 1, pos: 1058
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 1253
type: B, layer: 1, pos: 1253
type: A, layer: 1, pos: 1237
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1057
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1088
type: B, layer: 1, pos: 1088
type: A, layer: 1, pos: 1243
type: B, layer: 1, pos: 1243
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1216
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1072
type: B, layer: 1, pos: 1072
type: A, layer: 1, pos: 1269
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1232
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1275
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1197
type: B, layer: 1, pos: 1197
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1259
type: A, layer: 1, pos: 1259
type: B, layer: 1, pos: 1198
type: A, layer: 1, pos: 1198
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1041
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1199
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1056
type: A, layer: 1, pos: 1056
type: B, layer: 1, pos: 1059
type: A, layer: 1, pos: 1059
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1040
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1025
type: A, layer: 1, pos: 1025

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4864905, upper bound: 69.6232586
time: 156.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4864905, upper bound: 69.6232586
time: 110.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 267.35 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 267.35
Output dim: 4, lower bound: -69.4864905, upper bound: 69.6232586
IS_A2, status: Status.VERIFIED, split count: 1, time: 267.35
Output dim: 4, lower bound: -69.4864905, upper bound: 69.6232586

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 122.27 + 267.35 = 389.63 seconds
