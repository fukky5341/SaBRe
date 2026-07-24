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
execution time: IAR + RelationalAnalysis = 2.83 + 119.48 = 122.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -69.6625962, upper bound: 69.6625962

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1265

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6547449, upper bound: 69.6623892
time: 110.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6623892, upper bound: 69.6547449
time: 130.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 240.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 240.81
Output dim: 4, lower bound: -69.6547449, upper bound: 69.6623892
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 240.81
Output dim: 4, lower bound: -69.6623892, upper bound: 69.6547449

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

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1703

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6309698, upper bound: 69.6520437
time: 137.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6444000, upper bound: 69.6386167
time: 137.28 seconds

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

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1236

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6549067, upper bound: 69.6544955
time: 111.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6621407, upper bound: 69.6472248
time: 160.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 273.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 273.92
Output dim: 4, lower bound: -69.6309698, upper bound: 69.6520437
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 273.92
Output dim: 4, lower bound: -69.6444000, upper bound: 69.6386167
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 273.92
Output dim: 4, lower bound: -69.6549067, upper bound: 69.6544955
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 273.92
Output dim: 4, lower bound: -69.6621407, upper bound: 69.6472248

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

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1057

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1276

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6206054, upper bound: 69.6518780
time: 135.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6308039, upper bound: 69.6416714
time: 103.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1276

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 694

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6313735, upper bound: 69.6377603
time: 207.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6435480, upper bound: 69.6257261
time: 102.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1260

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6451140, upper bound: 69.6543862
time: 202.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6547957, upper bound: 69.6447098
time: 111.33 seconds

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

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6619379, upper bound: 69.6276848
time: 112.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6425903, upper bound: 69.6470236
time: 105.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 219.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6206054, upper bound: 69.6518780
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6308039, upper bound: 69.6416714
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6313735, upper bound: 69.6377603
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6435480, upper bound: 69.6257261
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6451140, upper bound: 69.6543862
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6547957, upper bound: 69.6447098
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6619379, upper bound: 69.6276848
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 219.92
Output dim: 4, lower bound: -69.6425903, upper bound: 69.6470236

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

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1271

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.6201352, upper bound: 69.5970585
time: 118.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5657594, upper bound: 69.6514078
time: 152.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1073

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5875273, upper bound: 69.6413204
time: 119.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6304532, upper bound: 69.5984153
time: 143.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1202

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6305173, upper bound: 69.6377351
time: 264.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6313476, upper bound: 69.6369131
time: 141.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6302759, upper bound: 69.6256973
time: 1761.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6435190, upper bound: 69.6124713
time: 109.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 696

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6328119, upper bound: 69.6542934
time: 115.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6450182, upper bound: 69.6497502
time: 115.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1559

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6374000, upper bound: 69.6422368
time: 140.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6523279, upper bound: 69.6273012
time: 113.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1720

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6614441, upper bound: 69.5772198
time: 149.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.6127120, upper bound: 69.6271344
time: 115.89 seconds

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6422506, upper bound: 69.6401293
time: 149.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6357393, upper bound: 69.6466830
time: 167.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 319.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6201352, upper bound: 69.5970585
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.5657594, upper bound: 69.6514078
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.5875273, upper bound: 69.6413204
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6304532, upper bound: 69.5984153
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6305173, upper bound: 69.6377351
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6313476, upper bound: 69.6369131
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6302759, upper bound: 69.6256973
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6435190, upper bound: 69.6124713
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6328119, upper bound: 69.6542934
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6450182, upper bound: 69.6497502
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6374000, upper bound: 69.6422368
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6523279, upper bound: 69.6273012
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6614441, upper bound: 69.5772198
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6127120, upper bound: 69.6271344
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6422506, upper bound: 69.6401293
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 319.15
Output dim: 4, lower bound: -69.6357393, upper bound: 69.6466830

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5657346, upper bound: 69.5951903
time: 105.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5155636, upper bound: 69.6513865
time: 142.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1059

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1592

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5874106, upper bound: 69.6402295
time: 172.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5864639, upper bound: 69.6412034
time: 142.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 661

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6301143, upper bound: 69.5915243
time: 113.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.6235740, upper bound: 69.5980766
time: 125.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1277

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 661

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5927139, upper bound: 69.6306253
time: 144.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5794403, upper bound: 69.5999965
time: 93.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1227

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6303248, upper bound: 69.6252738
time: 106.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6197365, upper bound: 69.6358914
time: 142.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 2.16 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 122.31 + 7078.03 = 7200.34 seconds
