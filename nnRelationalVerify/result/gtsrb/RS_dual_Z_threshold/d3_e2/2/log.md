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
execution time: IAR + RelationalAnalysis = 2.94 + 117.41 = 120.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -69.6625962, upper bound: 69.6625962

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6047801, upper bound: 69.6616908
time: 114.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6616908, upper bound: 69.6047801
time: 122.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 237.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 237.61
Output dim: 4, lower bound: -69.6047801, upper bound: 69.6616908
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 237.61
Output dim: 4, lower bound: -69.6616908, upper bound: 69.6047801

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5610066, upper bound: 69.6579973
time: 124.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.6005334, upper bound: 69.5919691
time: 166.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5919691, upper bound: 69.6005334
time: 155.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6579973, upper bound: 69.5610066
time: 132.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 290.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 290.51
Output dim: 4, lower bound: -69.5610066, upper bound: 69.6579973
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 290.51
Output dim: 4, lower bound: -69.6005334, upper bound: 69.5919691
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 290.51
Output dim: 4, lower bound: -69.5919691, upper bound: 69.6005334
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 290.51
Output dim: 4, lower bound: -69.6579973, upper bound: 69.5610066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4968697, upper bound: 69.6574096
time: 148.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5603031, upper bound: 69.5879672
time: 99.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5879672, upper bound: 69.5603031
time: 98.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6574096, upper bound: 69.4968697
time: 119.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 220.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 220.18
Output dim: 4, lower bound: -69.4968697, upper bound: 69.6574096
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 220.18
Output dim: 4, lower bound: -69.5603031, upper bound: 69.5879672
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 220.18
Output dim: 4, lower bound: -69.5879672, upper bound: 69.5603031
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 220.18
Output dim: 4, lower bound: -69.6574096, upper bound: 69.4968697

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4094795, upper bound: 69.6557083
time: 124.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4951197, upper bound: 69.5724451
time: 96.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5724451, upper bound: 69.4951197
time: 134.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6557083, upper bound: 69.4094795
time: 100.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 237.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 237.29
Output dim: 4, lower bound: -69.4094795, upper bound: 69.6557083
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 237.29
Output dim: 4, lower bound: -69.4951197, upper bound: 69.5724451
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 237.29
Output dim: 4, lower bound: -69.5724451, upper bound: 69.4951197
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 237.29
Output dim: 4, lower bound: -69.6557083, upper bound: 69.4094795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3786709, upper bound: 69.6480362
time: 118.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4007403, upper bound: 69.6128750
time: 178.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.6128750, upper bound: 69.4007403
time: 276.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5787987, upper bound: 69.3786709
time: 122.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 402.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 402.40
Output dim: 4, lower bound: -69.3786709, upper bound: 69.6480362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 402.40
Output dim: 4, lower bound: -69.4007403, upper bound: 69.6128750
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 402.40
Output dim: 4, lower bound: -69.6128750, upper bound: 69.4007403
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 402.40
Output dim: 4, lower bound: -69.5787987, upper bound: 69.3786709

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3439513, upper bound: 69.6465998
time: 132.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.3745649, upper bound: 69.5845505
time: 231.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 367.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 367.05
Output dim: 4, lower bound: -69.3439513, upper bound: 69.6465998
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 367.05
Output dim: 4, lower bound: -69.3745649, upper bound: 69.5845505

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3432651, upper bound: 69.6454091
time: 101.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3425727, upper bound: 69.6457817
time: 174.84 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 279.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 279.13
Output dim: 4, lower bound: -69.3432651, upper bound: 69.6454091
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 279.13
Output dim: 4, lower bound: -69.3425727, upper bound: 69.6457817

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.3423910, upper bound: 69.6100975
time: 124.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3076425, upper bound: 69.6445630
time: 118.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.3416953, upper bound: 69.6104786
time: 119.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3069126, upper bound: 69.6449349
time: 106.90 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 228.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 228.38
Output dim: 4, lower bound: -69.3423910, upper bound: 69.6100975
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 228.38
Output dim: 4, lower bound: -69.3076425, upper bound: 69.6445630
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 228.38
Output dim: 4, lower bound: -69.3416953, upper bound: 69.6104786
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 228.38
Output dim: 4, lower bound: -69.3069126, upper bound: 69.6449349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 695

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2873322, upper bound: 69.6431930
time: 103.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.3063141, upper bound: 69.6266232
time: 109.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 695

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2878176, upper bound: 69.6436463
time: 115.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.3055363, upper bound: 69.6259581
time: 120.25 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 237.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 237.92
Output dim: 4, lower bound: -69.2873322, upper bound: 69.6431930
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 237.92
Output dim: 4, lower bound: -69.3063141, upper bound: 69.6266232
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 237.92
Output dim: 4, lower bound: -69.2878176, upper bound: 69.6436463
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 237.92
Output dim: 4, lower bound: -69.3055363, upper bound: 69.6259581

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2855102, upper bound: 69.6312567
time: 207.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2810233, upper bound: 69.6419524
time: 121.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2859936, upper bound: 69.6316992
time: 104.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2815033, upper bound: 69.6424035
time: 98.55 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 205.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 205.86
Output dim: 4, lower bound: -69.2855102, upper bound: 69.6312567
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 205.86
Output dim: 4, lower bound: -69.2810233, upper bound: 69.6419524
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 205.86
Output dim: 4, lower bound: -69.2859936, upper bound: 69.6316992
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 205.86
Output dim: 4, lower bound: -69.2815033, upper bound: 69.6424035

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2850135, upper bound: 69.6065747
time: 128.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2595180, upper bound: 69.6307818
time: 133.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2805217, upper bound: 69.6158492
time: 104.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2551447, upper bound: 69.6414807
time: 113.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2854967, upper bound: 69.6070261
time: 129.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2599733, upper bound: 69.6312239
time: 120.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2809972, upper bound: 69.6162995
time: 120.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2555997, upper bound: 69.6419321
time: 120.85 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 243.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2850135, upper bound: 69.6065747
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2595180, upper bound: 69.6307818
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2805217, upper bound: 69.6158492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2551447, upper bound: 69.6414807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2854967, upper bound: 69.6070261
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2599733, upper bound: 69.6312239
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2809972, upper bound: 69.6162995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 243.47
Output dim: 4, lower bound: -69.2555997, upper bound: 69.6419321

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1595

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2593481, upper bound: 69.5893274
time: 107.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2176899, upper bound: 69.6306144
time: 109.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1595

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2549782, upper bound: 69.5997282
time: 115.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2137511, upper bound: 69.6412722
time: 230.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1595

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2598058, upper bound: 69.5897878
time: 133.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2181541, upper bound: 69.6310563
time: 111.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1595

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2554332, upper bound: 69.6002106
time: 105.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.2142129, upper bound: 69.6417234
time: 127.73 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 235.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2593481, upper bound: 69.5893274
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2176899, upper bound: 69.6306144
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2549782, upper bound: 69.5997282
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2137511, upper bound: 69.6412722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2598058, upper bound: 69.5897878
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2181541, upper bound: 69.6310563
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2554332, upper bound: 69.6002106
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 235.60
Output dim: 4, lower bound: -69.2142129, upper bound: 69.6417234

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -100.2228928, 48.5793381, -100.2228928, 48.5793381, -148.8022308, 148.8022308
1: -56.6546211, 47.4276047, -56.6546211, 47.4276047, -104.0822296, 104.0822296
2: -48.7933884, 44.5032082, -48.7933884, 44.5032082, -93.2966003, 93.2965927
3: -53.9330940, 56.4116364, -53.9330940, 56.4116364, -110.3447266, 110.3447266
4: -57.8086700, 55.3942719, -57.8086700, 55.3942719, -113.2029419, 113.2029419
5: -52.3867950, 57.4174957, -52.3867950, 57.4174957, -109.8042908, 109.8042908
6: -70.2205811, 50.7839775, -70.2205811, 50.7839775, -121.0045547, 121.0045624
7: -64.9685059, 59.3739700, -64.9685059, 59.3739700, -124.3424759, 124.3424759
8: -64.0200729, 68.8925018, -64.0200729, 68.8925018, -132.9125671, 132.9125671
9: -59.4585571, 51.1002731, -59.4585571, 51.1002731, -110.5588303, 110.5588303
10: -83.7690125, 77.7103348, -83.7690125, 77.7103348, -161.4793396, 161.4793396
11: -78.8429337, 60.4357147, -78.8429337, 60.4357147, -139.2786560, 139.2786407
12: -77.5734406, 65.4165649, -77.5734406, 65.4165649, -142.9899750, 142.9899902
13: -80.1185608, 73.5702744, -80.1185608, 73.5702744, -153.6888428, 153.6888275
14: -122.0133972, 61.5194054, -122.0133972, 61.5194054, -183.5328064, 183.5328064
15: -66.4472733, 49.4722939, -66.4472733, 49.4722939, -115.9195709, 115.9195709
16: -87.9878845, 60.0501404, -87.9878845, 60.0501404, -148.0380249, 148.0380249
17: -123.1345215, 92.8631134, -123.1345215, 92.8631134, -215.9976196, 215.9976349
18: -70.7084808, 59.8530006, -70.7084808, 59.8530006, -130.5614777, 130.5614777
19: -55.5216026, 35.5992279, -55.5216026, 35.5992279, -91.1208267, 91.1208344
20: -50.6315765, 42.8828735, -50.6315765, 42.8828735, -93.5144501, 93.5144348
21: -71.5156097, 46.3427963, -71.5156097, 46.3427963, -117.8583984, 117.8584061
22: -78.0901718, 48.5186844, -78.0901718, 48.5186844, -126.6088562, 126.6088562
23: -56.7396088, 45.3335419, -56.7396088, 45.3335419, -102.0731506, 102.0731506
24: -70.8121185, 51.9838791, -70.8121185, 51.9838791, -122.7959976, 122.7959976
25: -58.6934395, 52.4420700, -58.6934395, 52.4420700, -111.1355133, 111.1355133
26: -77.9699249, 70.1775208, -77.9699249, 70.1775208, -148.1474457, 148.1474457
27: -74.7296829, 51.2473450, -74.7296829, 51.2473450, -125.9770279, 125.9770279
28: -55.4958878, 48.9827728, -55.4958878, 48.9827728, -104.4786606, 104.4786530
29: -89.4714737, 50.0727081, -89.4714737, 50.0727081, -139.5441742, 139.5441895
30: -67.8267822, 60.8395538, -67.8267822, 60.8395538, -128.6663208, 128.6663361
31: -72.0573730, 48.3828125, -72.0573730, 48.3828125, -120.4401855, 120.4401855
32: -75.2747116, 48.5530510, -75.2747116, 48.5530510, -123.8277588, 123.8277588
33: -98.1314392, 69.1865540, -98.1314392, 69.1865540, -167.3179932, 167.3179932
34: -81.6724091, 54.4443550, -81.6724091, 54.4443550, -136.1167603, 136.1167603
35: -84.3818207, 57.2352638, -84.3818207, 57.2352638, -141.6170807, 141.6170807
36: -82.3326492, 56.4487000, -82.3326492, 56.4487000, -138.7813416, 138.7813416
37: -114.9885178, 58.2267838, -114.9885178, 58.2267838, -173.2153015, 173.2153015
38: -100.5695953, 68.7171631, -100.5695953, 68.7171631, -169.2867432, 169.2867432
39: -117.6973801, 67.3298874, -117.6973801, 67.3298874, -185.0272675, 185.0272675
40: -97.6037598, 53.3045845, -97.6037598, 53.3045845, -150.9083405, 150.9083405
41: -72.7334137, 46.9699516, -72.7334137, 46.9699516, -119.7033615, 119.7033691
42: -54.5271912, 43.7154465, -54.5271912, 43.7154465, -98.2426376, 98.2426376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=188, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2008047, upper bound: 69.6258718
time: 769.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.2129180, upper bound: 69.6126907
time: 121.00 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 892.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 892.76
Output dim: 4, lower bound: -69.2008047, upper bound: 69.6258718
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 892.76
Output dim: 4, lower bound: -69.2129180, upper bound: 69.6126907
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 892.76
Output dim: 4, lower bound: -69.2137511, upper bound: 69.6412722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 892.76
Output dim: 4, lower bound: -69.2181541, upper bound: 69.6310563
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 892.76
Output dim: 4, lower bound: -69.2142129, upper bound: 69.6417234

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 120.35 + 7489.27 = 7609.63 seconds
