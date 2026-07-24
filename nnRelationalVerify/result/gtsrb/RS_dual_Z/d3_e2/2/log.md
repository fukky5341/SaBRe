## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.74 + 119.23 = 121.96 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6047801, upper bound: 69.6616908
time: 115.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6616908, upper bound: 69.6047801
time: 115.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 230.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 230.48
Output dim: 4, lower bound: -69.6047801, upper bound: 69.6616908
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 230.48
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

Time for backsubstitution: 2.13 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5610066, upper bound: 69.6579973
time: 125.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6005334, upper bound: 69.5919691
time: 157.69 seconds

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

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5919691, upper bound: 69.6005334
time: 150.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6579973, upper bound: 69.5610066
time: 125.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 278.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 278.04
Output dim: 4, lower bound: -69.5610066, upper bound: 69.6579973
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 278.04
Output dim: 4, lower bound: -69.6005334, upper bound: 69.5919691
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 278.04
Output dim: 4, lower bound: -69.5919691, upper bound: 69.6005334
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 278.04
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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4968697, upper bound: 69.6574096
time: 145.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5603031, upper bound: 69.5879672
time: 99.78 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5435267, upper bound: 69.5913493
time: 106.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5435267, upper bound: 69.5214231
time: 133.35 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5214231, upper bound: 69.5997847
time: 179.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5913493, upper bound: 69.5435267
time: 114.36 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5879672, upper bound: 69.5603031
time: 99.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6574096, upper bound: 69.4968697
time: 122.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 224.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.4968697, upper bound: 69.6574096
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5603031, upper bound: 69.5879672
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5435267, upper bound: 69.5913493
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5435267, upper bound: 69.5214231
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5214231, upper bound: 69.5997847
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5913493, upper bound: 69.5435267
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 224.50
Output dim: 4, lower bound: -69.5879672, upper bound: 69.5603031
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 224.50
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

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4094795, upper bound: 69.6557083
time: 125.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4951197, upper bound: 69.5724451
time: 94.66 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4741039, upper bound: 69.5862371
time: 143.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5586080, upper bound: 69.5022109
time: 104.34 seconds

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

Time for backsubstitution: 2.13 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4570693, upper bound: 69.5896596
time: 99.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5417386, upper bound: 69.5053416
time: 117.07 seconds

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

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5142726, upper bound: 69.5196941
time: 127.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5417386, upper bound: 69.4345015
time: 125.50 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4345016, upper bound: 69.5980559
time: 136.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5196941, upper bound: 69.5142726
time: 138.71 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5053416, upper bound: 69.5417386
time: 114.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5896596, upper bound: 69.4570693
time: 111.91 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5022109, upper bound: 69.5586080
time: 120.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5862371, upper bound: 69.4741039
time: 148.10 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5724451, upper bound: 69.4951197
time: 134.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6557083, upper bound: 69.4094795
time: 98.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 235.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.4094795, upper bound: 69.6557083
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.4951197, upper bound: 69.5724451
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.4741039, upper bound: 69.5862371
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5586080, upper bound: 69.5022109
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.4570693, upper bound: 69.5896596
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5417386, upper bound: 69.5053416
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5142726, upper bound: 69.5196941
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5417386, upper bound: 69.4345015
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.4345016, upper bound: 69.5980559
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5196941, upper bound: 69.5142726
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5053416, upper bound: 69.5417386
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5896596, upper bound: 69.4570693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5022109, upper bound: 69.5586080
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5862371, upper bound: 69.4741039
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.29
Output dim: 4, lower bound: -69.5724451, upper bound: 69.4951197
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.29
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

Time for backsubstitution: 2.12 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.3786709, upper bound: 69.6480362
time: 114.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4007403, upper bound: 69.6128750
time: 174.57 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4644187, upper bound: 69.5646671
time: 114.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4863297, upper bound: 69.5288915
time: 148.34 seconds

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

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4424931, upper bound: 69.5787987
time: 108.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4652583, upper bound: 69.5433755
time: 102.25 seconds

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

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5273314, upper bound: 69.4946986
time: 142.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5499415, upper bound: 69.4587559
time: 119.29 seconds

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

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4135038, upper bound: 69.5807783
time: 128.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4497439, upper bound: 69.5629776
time: 151.43 seconds

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

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4987812, upper bound: 69.4963940
time: 95.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5344964, upper bound: 69.4784827
time: 120.30 seconds

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

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4704590, upper bound: 69.5107527
time: 173.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5067207, upper bound: 69.4929367
time: 114.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5547273, upper bound: 69.4254954
time: 126.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5905374, upper bound: 69.4076434
time: 113.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4076435, upper bound: 69.5905374
time: 190.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4254954, upper bound: 69.5547273
time: 140.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4929367, upper bound: 69.5067207
time: 108.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5107527, upper bound: 69.4704590
time: 115.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4784827, upper bound: 69.5344964
time: 130.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4963940, upper bound: 69.4987811
time: 114.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5629776, upper bound: 69.4497439
time: 137.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5807783, upper bound: 69.4135038
time: 126.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.22 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4587559, upper bound: 69.5499415
time: 120.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4946986, upper bound: 69.5273314
time: 180.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 303.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.3786709, upper bound: 69.6480362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4007403, upper bound: 69.6128750
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4644187, upper bound: 69.5646671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4863297, upper bound: 69.5288915
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4424931, upper bound: 69.5787987
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4652583, upper bound: 69.5433755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5273314, upper bound: 69.4946986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5499415, upper bound: 69.4587559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4135038, upper bound: 69.5807783
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4497439, upper bound: 69.5629776
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4987812, upper bound: 69.4963940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5344964, upper bound: 69.4784827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4704590, upper bound: 69.5107527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5067207, upper bound: 69.4929367
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5547273, upper bound: 69.4254954
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5905374, upper bound: 69.4076434
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4076435, upper bound: 69.5905374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4254954, upper bound: 69.5547273
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4929367, upper bound: 69.5067207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5107527, upper bound: 69.4704590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4784827, upper bound: 69.5344964
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4963940, upper bound: 69.4987811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5629776, upper bound: 69.4497439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.5807783, upper bound: 69.4135038
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4587559, upper bound: 69.5499415
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 303.26
Output dim: 4, lower bound: -69.4946986, upper bound: 69.5273314
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 303.26
Output dim: 4, lower bound: -69.5862371, upper bound: 69.4741039
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 303.26
Output dim: 4, lower bound: -69.5724451, upper bound: 69.4951197
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 303.26
Output dim: 4, lower bound: -69.6557083, upper bound: 69.4094795

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 121.96 + 7206.19 = 7328.16 seconds
