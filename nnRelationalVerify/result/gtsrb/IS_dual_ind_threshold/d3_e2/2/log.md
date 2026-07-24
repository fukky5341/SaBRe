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
execution time: IAR + RelationalAnalysis = 3.05 + 118.69 = 121.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -69.6625962, upper bound: 69.6625962

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1121

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
time: 271.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
time: 577.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 849.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 849.56
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
IS_A2, status: Status.UNKNOWN, split count: 1, time: 849.56
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -99.9720459, 48.4851074, -100.1434708, 48.5497475, -148.5217896, 148.6285706
1: -56.5383034, 47.2901535, -56.6189957, 47.3890762, -103.9273834, 103.9091492
2: -48.5665932, 44.4318008, -48.7243996, 44.4809189, -93.0475159, 93.1561966
3: -53.6739273, 56.3102913, -53.8548126, 56.3809891, -110.0549011, 110.1650925
4: -57.4597931, 55.2988892, -57.7040672, 55.3720474, -112.8318405, 113.0029602
5: -52.1733780, 57.3041458, -52.3218536, 57.3817635, -109.5551376, 109.6259995
6: -70.0861893, 50.4166412, -70.1858215, 50.6745834, -120.7607574, 120.6024551
7: -64.8497620, 59.1724930, -64.9254150, 59.3160858, -124.1658478, 124.0979080
8: -63.7219849, 68.7808380, -63.9297066, 68.8600769, -132.5820465, 132.7105408
9: -59.2283745, 50.9391785, -59.3895836, 51.0540886, -110.2824631, 110.3287659
10: -83.5794830, 77.3305359, -83.7139893, 77.5974655, -161.1769409, 161.0445251
11: -78.6325150, 59.7525635, -78.7940521, 60.2334290, -138.8659363, 138.5466156
12: -77.3877106, 64.7767181, -77.5345306, 65.2302551, -142.6179657, 142.3112488
13: -79.9936829, 73.1153107, -80.0826187, 73.4398041, -153.4334869, 153.1979370
14: -121.7941589, 60.9223328, -121.9602737, 61.3496323, -183.1437988, 182.8825989
15: -65.9452057, 49.4036674, -66.2929993, 49.4335060, -115.3787079, 115.6966705
16: -87.8031311, 59.6514816, -87.9242096, 59.9275055, -147.7306366, 147.5756836
17: -122.8538666, 91.8122711, -123.0809708, 92.5592422, -215.4130707, 214.8932495
18: -70.4102325, 59.7390633, -70.6227417, 59.8147354, -130.2249451, 130.3618011
19: -55.3677597, 35.3070526, -55.4833984, 35.5104065, -90.8781586, 90.7904434
20: -50.5101318, 42.6247025, -50.5987091, 42.8076859, -93.3178177, 93.2234039
21: -71.3475266, 45.9627457, -71.4740753, 46.2321548, -117.5796814, 117.4368134
22: -77.8689270, 48.2847748, -78.0264893, 48.4493637, -126.3182907, 126.3112640
23: -56.6209869, 45.1135712, -56.7092323, 45.2675018, -101.8884888, 101.8227997
24: -70.5814209, 51.9221916, -70.7439728, 51.9667664, -122.5481873, 122.6661682
25: -58.5443649, 52.1989365, -58.6507530, 52.3720398, -110.9164047, 110.8496857
26: -77.7713547, 69.9282532, -77.9156570, 70.1001892, -147.8715363, 147.8439026
27: -74.4797821, 51.1869164, -74.6546021, 51.2222214, -125.7020035, 125.8415222
28: -55.3755417, 48.7620010, -55.4662132, 48.9157410, -104.2912750, 104.2282104
29: -89.3361664, 49.7068253, -89.4346313, 49.9643860, -139.3005524, 139.1414490
30: -67.7197952, 60.5500717, -67.7876663, 60.7543869, -128.4741821, 128.3377380
31: -71.9006500, 48.1720810, -72.0059052, 48.3183670, -120.2190170, 120.1779785
32: -75.1715393, 48.2525291, -75.2448578, 48.4631119, -123.6346436, 123.4973755
33: -97.6944275, 69.0692139, -98.0051727, 69.1557159, -166.8501434, 167.0743866
34: -81.3240280, 54.3310318, -81.5722046, 54.4148140, -135.7388458, 135.9032288
35: -84.0802307, 57.1569366, -84.2939377, 57.2144852, -141.2947083, 141.4508667
36: -82.1658478, 56.2879753, -82.2852478, 56.4014549, -138.5673065, 138.5732117
37: -114.6117401, 58.1408768, -114.8796387, 58.2005424, -172.8122864, 173.0205078
38: -100.3190918, 68.5386124, -100.4967194, 68.6646042, -168.9836884, 169.0353394
39: -117.3436661, 67.2601929, -117.5935669, 67.3109665, -184.6546326, 184.8537292
40: -97.1583176, 53.1989746, -97.4744873, 53.2798462, -150.4381714, 150.6734619
41: -72.5600281, 46.7714386, -72.6830978, 46.9075699, -119.4675980, 119.4545288
42: -54.4054337, 43.3735962, -54.4967384, 43.6116180, -98.0170441, 97.8703308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=187, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=735, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1121

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5666756
time: 118.07 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5968062, upper bound: 69.6594478
time: 141.98 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -100.2133789, 48.5769882, -100.2228928, 48.5793381, -148.7927246, 148.7998810
1: -56.6496010, 47.4203453, -56.6546211, 47.4276047, -104.0772095, 104.0749664
2: -48.7867737, 44.5010529, -48.7933884, 44.5032082, -93.2899780, 93.2944412
3: -53.9270096, 56.4083099, -53.9330940, 56.4116364, -110.3386459, 110.3414001
4: -57.8006477, 55.3915558, -57.8086700, 55.3942719, -113.1949158, 113.2002258
5: -52.3790817, 57.4138107, -52.3867950, 57.4174957, -109.7965775, 109.8006058
6: -70.2161560, 50.7749023, -70.2205811, 50.7839775, -121.0001373, 120.9954834
7: -64.9640198, 59.3626785, -64.9685059, 59.3739700, -124.3379898, 124.3311844
8: -64.0129700, 68.8893433, -64.0200729, 68.8925018, -132.9054718, 132.9094238
9: -59.4515572, 51.0926132, -59.4585571, 51.1002731, -110.5518341, 110.5511627
10: -83.7617493, 77.7007599, -83.7690125, 77.7103348, -161.4720764, 161.4697723
11: -78.8379364, 60.4215965, -78.8429337, 60.4357147, -139.2736511, 139.2645264
12: -77.5697021, 65.4040375, -77.5734406, 65.4165649, -142.9862671, 142.9774780
13: -80.1138000, 73.5606079, -80.1185608, 73.5702744, -153.6840820, 153.6791687
14: -122.0064392, 61.5088577, -122.0133972, 61.5194054, -183.5258484, 183.5222473
15: -66.4320068, 49.4682465, -66.4472733, 49.4722939, -115.9042969, 115.9155121
16: -87.9804077, 60.0269012, -87.9878845, 60.0501404, -148.0305481, 148.0147858
17: -123.1297607, 92.8440170, -123.1345215, 92.8631134, -215.9928589, 215.9785156
18: -70.6920471, 59.8486748, -70.7084808, 59.8530006, -130.5450439, 130.5571442
19: -55.5178604, 35.5943375, -55.5216026, 35.5992279, -91.1170883, 91.1159363
20: -50.6286926, 42.8769264, -50.6315765, 42.8828735, -93.5115662, 93.5084991
21: -71.5113525, 46.3352242, -71.5156097, 46.3427963, -117.8541489, 117.8508301
22: -78.0781174, 48.5118103, -78.0901718, 48.5186844, -126.5968018, 126.6019821
23: -56.7367783, 45.3266869, -56.7396088, 45.3335419, -102.0703125, 102.0662994
24: -70.8049088, 51.9816895, -70.8121185, 51.9838791, -122.7887878, 122.7938080
25: -58.6886520, 52.4380302, -58.6934395, 52.4420700, -111.1307220, 111.1314697
26: -77.9576569, 70.1691742, -77.9699249, 70.1775208, -148.1351776, 148.1390991
27: -74.7180252, 51.2440720, -74.7296829, 51.2473450, -125.9653473, 125.9737549
28: -55.4932022, 48.9783173, -55.4958878, 48.9827728, -104.4759750, 104.4742050
29: -89.4667206, 50.0640640, -89.4714737, 50.0727081, -139.5394287, 139.5355377
30: -67.8220673, 60.8186874, -67.8267822, 60.8395538, -128.6616211, 128.6454468
31: -72.0523834, 48.3770828, -72.0573730, 48.3828125, -120.4351807, 120.4344559
32: -75.2709808, 48.5453568, -75.2747116, 48.5530510, -123.8240204, 123.8200684
33: -98.1208801, 69.1833649, -98.1314392, 69.1865540, -167.3074188, 167.3147888
34: -81.6650772, 54.4407387, -81.6724091, 54.4443550, -136.1094360, 136.1131439
35: -84.3739929, 57.2330589, -84.3818207, 57.2352638, -141.6092529, 141.6148834
36: -82.3243484, 56.4442749, -82.3326492, 56.4487000, -138.7730408, 138.7769165
37: -114.9787750, 58.2241440, -114.9885178, 58.2267838, -173.2055359, 173.2126617
38: -100.5520401, 68.7120667, -100.5695953, 68.7171631, -169.2691956, 169.2816620
39: -117.6877213, 67.3240509, -117.6973801, 67.3298874, -185.0176086, 185.0214233
40: -97.5934296, 53.3004608, -97.6037598, 53.3045845, -150.8980103, 150.9042206
41: -72.7253723, 46.9633675, -72.7334137, 46.9699516, -119.6953278, 119.6967773
42: -54.5235481, 43.7069016, -54.5271912, 43.7154465, -98.2389832, 98.2340927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=187, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6611616, upper bound: 69.5999827
time: 117.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6611616, upper bound: 69.6611616
time: 121.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 241.28 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 241.28
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5666756
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 241.28
Output dim: 4, lower bound: -69.5968062, upper bound: 69.6594478
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 241.28
Output dim: 4, lower bound: -69.6611616, upper bound: 69.5999827
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 241.28
Output dim: 4, lower bound: -69.6611616, upper bound: 69.6611616

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -99.9720459, 48.4851074, -100.1376343, 48.5483170, -148.5203552, 148.6227264
1: -56.5383034, 47.2901535, -56.6154099, 47.3880997, -103.9264069, 103.9055634
2: -48.5665932, 44.4318008, -48.7198715, 44.4795837, -93.0461731, 93.1516724
3: -53.6739273, 56.3102913, -53.8494644, 56.3788261, -110.0527496, 110.1597443
4: -57.4597931, 55.2988892, -57.6982574, 55.3707199, -112.8305130, 112.9971466
5: -52.1733780, 57.3041458, -52.3170471, 57.3794785, -109.5528564, 109.6211929
6: -70.0861893, 50.4166412, -70.1835022, 50.6677361, -120.7539215, 120.6001205
7: -64.8497620, 59.1724930, -64.9208908, 59.3147774, -124.1645355, 124.0933838
8: -63.7219849, 68.7808380, -63.9242783, 68.8583221, -132.5803070, 132.7051086
9: -59.2283745, 50.9391785, -59.3854599, 51.0504379, -110.2788086, 110.3246384
10: -83.5794830, 77.3305359, -83.7079010, 77.5917358, -161.1712036, 161.0384369
11: -78.6325150, 59.7525635, -78.7908783, 60.2252960, -138.8578033, 138.5434418
12: -77.3877106, 64.7767181, -77.5325928, 65.2214355, -142.6091461, 142.3092804
13: -79.9936829, 73.1153107, -80.0772781, 73.4322357, -153.4259186, 153.1925964
14: -121.7941589, 60.9223328, -121.9565430, 61.3446045, -183.1387634, 182.8788757
15: -65.9452057, 49.4036674, -66.2829285, 49.4309998, -115.3762054, 115.6865997
16: -87.8031311, 59.6514816, -87.9177628, 59.9094124, -147.7125397, 147.5692444
17: -122.8538666, 91.8122711, -123.0780792, 92.5485229, -215.4023895, 214.8903503
18: -70.4102325, 59.7390633, -70.6142273, 59.8107834, -130.2210083, 130.3532715
19: -55.3677597, 35.3070526, -55.4809990, 35.5062523, -90.8740082, 90.7880478
20: -50.5101318, 42.6247025, -50.5967789, 42.8039703, -93.3141022, 93.2214661
21: -71.3475266, 45.9627457, -71.4716339, 46.2268295, -117.5743561, 117.4343796
22: -77.8689270, 48.2847748, -78.0239105, 48.4438400, -126.3127670, 126.3086853
23: -56.6209869, 45.1135712, -56.7075958, 45.2642365, -101.8852234, 101.8211670
24: -70.5814209, 51.9221916, -70.7388000, 51.9634018, -122.5448227, 122.6609879
25: -58.5443649, 52.1989365, -58.6476593, 52.3685226, -110.9128876, 110.8465881
26: -77.7713547, 69.9282532, -77.9128647, 70.0938110, -147.8651581, 147.8411102
27: -74.4797821, 51.1869164, -74.6519318, 51.2174683, -125.6972504, 125.8388519
28: -55.3755417, 48.7620010, -55.4643745, 48.9123230, -104.2878571, 104.2263794
29: -89.3361664, 49.7068253, -89.4321747, 49.9572906, -139.2934570, 139.1390076
30: -67.7197952, 60.5500717, -67.7849884, 60.7450676, -128.4648590, 128.3350525
31: -71.9006500, 48.1720810, -72.0030136, 48.3159370, -120.2165833, 120.1750793
32: -75.1715393, 48.2525291, -75.2427979, 48.4575882, -123.6291199, 123.4953156
33: -97.6944275, 69.0692139, -97.9993286, 69.1538086, -166.8482361, 167.0685272
34: -81.3240280, 54.3310318, -81.5676422, 54.4128113, -135.7368469, 135.8986664
35: -84.0802307, 57.1569366, -84.2895355, 57.2131805, -141.2934113, 141.4464722
36: -82.1658478, 56.2879753, -82.2830353, 56.3979340, -138.5637817, 138.5710144
37: -114.6117401, 58.1408768, -114.8748703, 58.1937256, -172.8054657, 173.0157471
38: -100.3190918, 68.5386124, -100.4893646, 68.6603088, -168.9793701, 169.0279846
39: -117.3436661, 67.2601929, -117.5885468, 67.3038177, -184.6474915, 184.8487396
40: -97.1583176, 53.1989746, -97.4693527, 53.2768631, -150.4351807, 150.6683350
41: -72.5600281, 46.7714386, -72.6805267, 46.8979874, -119.4580154, 119.4519653
42: -54.4054337, 43.3735962, -54.4948540, 43.6064491, -98.0118866, 97.8684540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=187, inp2_unstable=187, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=735, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1121

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6561940
time: 138.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6594479
time: 95.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -100.2133789, 48.5769882, -99.9720459, 48.4851074, -148.6984863, 148.5490417
1: -56.6496010, 47.4203453, -56.5383034, 47.2901535, -103.9397583, 103.9586487
2: -48.7867737, 44.5010529, -48.5665932, 44.4318008, -93.2185669, 93.0676422
3: -53.9270096, 56.4083099, -53.6739273, 56.3102913, -110.2373047, 110.0822296
4: -57.8006477, 55.3915558, -57.4597931, 55.2988892, -113.0995331, 112.8513489
5: -52.3790817, 57.4138107, -52.1733780, 57.3041458, -109.6832199, 109.5871887
6: -70.2161560, 50.7749023, -70.0861893, 50.4166412, -120.6327972, 120.8610840
7: -64.9640198, 59.3626785, -64.8497620, 59.1724930, -124.1365128, 124.2124405
8: -64.0129700, 68.8893433, -63.7219849, 68.7808380, -132.7938080, 132.6113281
9: -59.4515572, 51.0926132, -59.2283745, 50.9391785, -110.3907318, 110.3209839
10: -83.7617493, 77.7007599, -83.5794830, 77.3305359, -161.0922852, 161.2802429
11: -78.8379364, 60.4215965, -78.6325150, 59.7525635, -138.5904999, 139.0541077
12: -77.5697021, 65.4040375, -77.3877106, 64.7767181, -142.3464203, 142.7917480
13: -80.1138000, 73.5606079, -79.9936829, 73.1153107, -153.2291107, 153.5542908
14: -122.0064392, 61.5088577, -121.7941589, 60.9223328, -182.9287720, 183.3029938
15: -66.4320068, 49.4682465, -65.9452057, 49.4036674, -115.8356781, 115.4134521
16: -87.9804077, 60.0269012, -87.8031311, 59.6514816, -147.6318970, 147.8300171
17: -123.1297607, 92.8440170, -122.8538666, 91.8122711, -214.9420319, 215.6978760
18: -70.6920471, 59.8486748, -70.4102325, 59.7390633, -130.4310913, 130.2588959
19: -55.5178604, 35.5943375, -55.3677597, 35.3070526, -90.8249130, 90.9620972
20: -50.6286926, 42.8769264, -50.5101318, 42.6247025, -93.2533951, 93.3870544
21: -71.5113525, 46.3352242, -71.3475266, 45.9627457, -117.4740982, 117.6827545
22: -78.0781174, 48.5118103, -77.8689270, 48.2847748, -126.3628922, 126.3807373
23: -56.7367783, 45.3266869, -56.6209869, 45.1135712, -101.8503494, 101.9476624
24: -70.8049088, 51.9816895, -70.5814209, 51.9221916, -122.7270966, 122.5631104
25: -58.6886520, 52.4380302, -58.5443649, 52.1989365, -110.8875885, 110.9823914
26: -77.9576569, 70.1691742, -77.7713547, 69.9282532, -147.8859100, 147.9405212
27: -74.7180252, 51.2440720, -74.4797821, 51.1869164, -125.9049377, 125.7238388
28: -55.4932022, 48.9783173, -55.3755417, 48.7620010, -104.2552032, 104.3538589
29: -89.4667206, 50.0640640, -89.3361664, 49.7068253, -139.1735382, 139.4002380
30: -67.8220673, 60.8186874, -67.7197952, 60.5500717, -128.3721313, 128.5384827
31: -72.0523834, 48.3770828, -71.9006500, 48.1720810, -120.2244492, 120.2777328
32: -75.2709808, 48.5453568, -75.1715393, 48.2525291, -123.5234985, 123.7168808
33: -98.1208801, 69.1833649, -97.6944275, 69.0692139, -167.1900635, 166.8777924
34: -81.6650772, 54.4407387, -81.3240280, 54.3310318, -135.9961090, 135.7647705
35: -84.3739929, 57.2330589, -84.0802307, 57.1569366, -141.5309296, 141.3132935
36: -82.3243484, 56.4442749, -82.1658478, 56.2879753, -138.6123199, 138.6101074
37: -114.9787750, 58.2241440, -114.6117401, 58.1408768, -173.1196594, 172.8358765
38: -100.5520401, 68.7120667, -100.3190918, 68.5386124, -169.0906372, 169.0311584
39: -117.6877213, 67.3240509, -117.3436661, 67.2601929, -184.9479065, 184.6676941
40: -97.5934296, 53.3004608, -97.1583176, 53.1989746, -150.7924042, 150.4587708
41: -72.7253723, 46.9633675, -72.5600281, 46.7714386, -119.4968109, 119.5233917
42: -54.5235481, 43.7069016, -54.4054337, 43.3735962, -97.8971405, 98.1123276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=187, inp2_unstable=187, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=735, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1121

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5938845
time: 111.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5968062
time: 170.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -100.2133789, 48.5769882, -100.2133789, 48.5769882, -148.7903748, 148.7903748
1: -56.6496010, 47.4203453, -56.6496010, 47.4203453, -104.0699463, 104.0699463
2: -48.7867737, 44.5010529, -48.7867737, 44.5010529, -93.2878189, 93.2878265
3: -53.9270096, 56.4083099, -53.9270096, 56.4083099, -110.3353195, 110.3353195
4: -57.8006477, 55.3915558, -57.8006477, 55.3915558, -113.1921997, 113.1921997
5: -52.3790817, 57.4138107, -52.3790817, 57.4138107, -109.7928772, 109.7928848
6: -70.2161560, 50.7749023, -70.2161560, 50.7749023, -120.9910583, 120.9910583
7: -64.9640198, 59.3626785, -64.9640198, 59.3626785, -124.3266983, 124.3266983
8: -64.0129700, 68.8893433, -64.0129700, 68.8893433, -132.9023132, 132.9023132
9: -59.4515572, 51.0926132, -59.4515572, 51.0926132, -110.5441666, 110.5441742
10: -83.7617493, 77.7007599, -83.7617493, 77.7007599, -161.4625092, 161.4625092
11: -78.8379364, 60.4215965, -78.8379364, 60.4215965, -139.2595367, 139.2595367
12: -77.5697021, 65.4040375, -77.5697021, 65.4040375, -142.9737396, 142.9737396
13: -80.1138000, 73.5606079, -80.1138000, 73.5606079, -153.6744080, 153.6744080
14: -122.0064392, 61.5088577, -122.0064392, 61.5088577, -183.5152893, 183.5152893
15: -66.4320068, 49.4682465, -66.4320068, 49.4682465, -115.9002457, 115.9002533
16: -87.9804077, 60.0269012, -87.9804077, 60.0269012, -148.0072937, 148.0072937
17: -123.1297607, 92.8440170, -123.1297607, 92.8440170, -215.9737549, 215.9737549
18: -70.6920471, 59.8486748, -70.6920471, 59.8486748, -130.5407104, 130.5407104
19: -55.5178604, 35.5943375, -55.5178604, 35.5943375, -91.1121979, 91.1121979
20: -50.6286926, 42.8769264, -50.6286926, 42.8769264, -93.5056152, 93.5056152
21: -71.5113525, 46.3352242, -71.5113525, 46.3352242, -117.8465729, 117.8465729
22: -78.0781174, 48.5118103, -78.0781174, 48.5118103, -126.5899277, 126.5899277
23: -56.7367783, 45.3266869, -56.7367783, 45.3266869, -102.0634613, 102.0634613
24: -70.8049088, 51.9816895, -70.8049088, 51.9816895, -122.7865982, 122.7865982
25: -58.6886520, 52.4380302, -58.6886520, 52.4380302, -111.1266785, 111.1266785
26: -77.9576569, 70.1691742, -77.9576569, 70.1691742, -148.1268311, 148.1268311
27: -74.7180252, 51.2440720, -74.7180252, 51.2440720, -125.9620819, 125.9620895
28: -55.4932022, 48.9783173, -55.4932022, 48.9783173, -104.4715195, 104.4715118
29: -89.4667206, 50.0640640, -89.4667206, 50.0640640, -139.5307922, 139.5307922
30: -67.8220673, 60.8186874, -67.8220673, 60.8186874, -128.6407318, 128.6407471
31: -72.0523834, 48.3770828, -72.0523834, 48.3770828, -120.4294662, 120.4294662
32: -75.2709808, 48.5453568, -75.2709808, 48.5453568, -123.8163376, 123.8163300
33: -98.1208801, 69.1833649, -98.1208801, 69.1833649, -167.3042297, 167.3042297
34: -81.6650772, 54.4407387, -81.6650772, 54.4407387, -136.1058197, 136.1058197
35: -84.3739929, 57.2330589, -84.3739929, 57.2330589, -141.6070557, 141.6070404
36: -82.3243484, 56.4442749, -82.3243484, 56.4442749, -138.7686157, 138.7686157
37: -114.9787750, 58.2241440, -114.9787750, 58.2241440, -173.2029114, 173.2029114
38: -100.5520401, 68.7120667, -100.5520401, 68.7120667, -169.2640991, 169.2640991
39: -117.6877213, 67.3240509, -117.6877213, 67.3240509, -185.0117645, 185.0117645
40: -97.5934296, 53.3004608, -97.5934296, 53.3004608, -150.8938904, 150.8938904
41: -72.7253723, 46.9633675, -72.7253723, 46.9633675, -119.6887360, 119.6887360
42: -54.5235481, 43.7069016, -54.5235481, 43.7069016, -98.2304382, 98.2304382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=187, inp2_unstable=187, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=734, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1121

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5666755, upper bound: 69.5946254
time: 112.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5978846
time: 124.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 239.57 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6561940
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6594479
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5938845
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5968062
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5666755, upper bound: 69.5946254
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 239.57
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5978846

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -99.6422348, 48.4110603, -100.1376343, 48.5483170, -148.1905518, 148.5486908
1: -56.3389130, 47.2463036, -56.6154099, 47.3880997, -103.7270050, 103.8617096
2: -48.2806168, 44.3637619, -48.7198715, 44.4795837, -92.7602005, 93.0836334
3: -53.3354645, 56.2194977, -53.8494644, 56.3788261, -109.7142944, 110.0689545
4: -57.1018524, 55.2226562, -57.6982574, 55.3707199, -112.4725647, 112.9208984
5: -51.8756332, 57.2097778, -52.3170471, 57.3794785, -109.2551117, 109.5268250
6: -69.9912796, 50.1353760, -70.1835022, 50.6677361, -120.6590118, 120.3188782
7: -64.6530609, 59.1130905, -64.9208908, 59.3147774, -123.9678345, 124.0339813
8: -63.3857040, 68.6983566, -63.9242783, 68.8583221, -132.2440186, 132.6226196
9: -59.0058174, 50.8339119, -59.3854599, 51.0504379, -110.0562515, 110.2193756
10: -83.4045258, 77.0621796, -83.7079010, 77.5917358, -160.9962463, 160.7700806
11: -78.4965744, 59.1900139, -78.7908783, 60.2252960, -138.7218628, 137.9808960
12: -77.2656708, 64.1919861, -77.5325928, 65.2214355, -142.4870911, 141.7245636
13: -79.8949814, 72.9078064, -80.0772781, 73.4322357, -153.3272095, 152.9850769
14: -121.6271667, 60.5731506, -121.9565430, 61.3446045, -182.9717712, 182.5296936
15: -65.5419922, 49.3656845, -66.2829285, 49.4309998, -114.9729919, 115.6486130
16: -87.6220703, 59.3837051, -87.9177628, 59.9094124, -147.5314789, 147.3014679
17: -122.6604767, 91.0631104, -123.0780792, 92.5485229, -215.2089996, 214.1411896
18: -70.2627640, 59.5181656, -70.6142273, 59.8107834, -130.0735474, 130.1323853
19: -55.2506409, 35.0257835, -55.4809990, 35.5062523, -90.7568893, 90.5067825
20: -50.4093399, 42.3887787, -50.5967789, 42.8039703, -93.2133026, 92.9855499
21: -71.2238770, 45.5859070, -71.4716339, 46.2268295, -117.4507065, 117.0575333
22: -77.7593842, 47.9861069, -78.0239105, 48.4438400, -126.2032242, 126.0100174
23: -56.5302887, 44.9199371, -56.7075958, 45.2642365, -101.7945251, 101.6275330
24: -70.4662933, 51.8561401, -70.7388000, 51.9634018, -122.4296799, 122.5949326
25: -58.4523201, 52.0072784, -58.6476593, 52.3685226, -110.8208389, 110.6549377
26: -77.6328659, 69.5193329, -77.9128647, 70.0938110, -147.7266846, 147.4321899
27: -74.3629074, 51.0571251, -74.6519318, 51.2174683, -125.5803757, 125.7090607
28: -55.2771492, 48.5479431, -55.4643745, 48.9123230, -104.1894684, 104.0123138
29: -89.2277832, 49.2786942, -89.4321747, 49.9572906, -139.1850739, 138.7108765
30: -67.6430664, 60.2611618, -67.7849884, 60.7450676, -128.3881226, 128.0461426
31: -71.7763367, 47.9661102, -72.0030136, 48.3159370, -120.0922699, 119.9691238
32: -75.0932159, 47.9593964, -75.2427979, 48.4575882, -123.5508041, 123.2021790
33: -97.3785782, 68.9677277, -97.9993286, 69.1538086, -166.5323792, 166.9670563
34: -81.0367661, 54.2382431, -81.5676422, 54.4128113, -135.4495850, 135.8058777
35: -83.8510590, 57.0984535, -84.2895355, 57.2131805, -141.0642395, 141.3879700
36: -82.0687637, 56.0769424, -82.2830353, 56.3979340, -138.4667053, 138.3599854
37: -114.4458008, 57.9527664, -114.8748703, 58.1937256, -172.6395264, 172.8276215
38: -100.0925293, 68.4101562, -100.4893646, 68.6603088, -168.7528076, 168.8995056
39: -117.1203003, 67.1925964, -117.5885468, 67.3038177, -184.4241028, 184.7811432
40: -96.9475098, 53.1239548, -97.4693527, 53.2768631, -150.2243652, 150.5932922
41: -72.4517670, 46.5846481, -72.6805267, 46.8979874, -119.3497543, 119.2651749
42: -54.3113708, 43.0742340, -54.4948540, 43.6064491, -97.9178162, 97.5690918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=186, inp2_unstable=187, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=735, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1121

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
time: 114.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
time: 117.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -99.9662552, 48.4835434, -100.1376343, 48.5483170, -148.5145721, 148.6211853
1: -56.5347748, 47.2891769, -56.6154099, 47.3880997, -103.9228668, 103.9045868
2: -48.5622482, 44.4304543, -48.7198715, 44.4795837, -93.0418243, 93.1503296
3: -53.6687622, 56.3080788, -53.8494644, 56.3788261, -110.0475769, 110.1575470
4: -57.4542198, 55.2975311, -57.6982574, 55.3707199, -112.8249359, 112.9957886
5: -52.1687775, 57.3017807, -52.3170471, 57.3794785, -109.5482559, 109.6188278
6: -70.0837708, 50.4099388, -70.1835022, 50.6677361, -120.7515106, 120.5934296
7: -64.8451538, 59.1712074, -64.9208908, 59.3147774, -124.1599197, 124.0920944
8: -63.7167282, 68.7790527, -63.9242783, 68.8583221, -132.5750427, 132.7033234
9: -59.2234421, 50.9355545, -59.3854599, 51.0504379, -110.2738800, 110.3210144
10: -83.5734787, 77.3249969, -83.7079010, 77.5917358, -161.1651917, 161.0328979
11: -78.6290283, 59.7447052, -78.7908783, 60.2252960, -138.8543243, 138.5355835
12: -77.3857269, 64.7681885, -77.5325928, 65.2214355, -142.6071625, 142.3007660
13: -79.9884872, 73.1077118, -80.0772781, 73.4322357, -153.4207153, 153.1849823
14: -121.7904892, 60.9173088, -121.9565430, 61.3446045, -183.1351013, 182.8738403
15: -65.9314423, 49.4010239, -66.2829285, 49.4309998, -115.3624420, 115.6839523
16: -87.7965546, 59.6336212, -87.9177628, 59.9094124, -147.7059631, 147.5513916
17: -122.8510056, 91.8015747, -123.0780792, 92.5485229, -215.3995361, 214.8796539
18: -70.4017334, 59.7351608, -70.6142273, 59.8107834, -130.2125244, 130.3493652
19: -55.3653107, 35.3030167, -55.4809990, 35.5062523, -90.8715668, 90.7840118
20: -50.5081139, 42.6210480, -50.5967789, 42.8039703, -93.3120728, 93.2178268
21: -71.3450317, 45.9575195, -71.4716339, 46.2268295, -117.5718536, 117.4291534
22: -77.8663635, 48.2790604, -78.0239105, 48.4438400, -126.3102036, 126.3029709
23: -56.6193657, 45.1103554, -56.7075958, 45.2642365, -101.8836060, 101.8179474
24: -70.5762634, 51.9191780, -70.7388000, 51.9634018, -122.5396500, 122.6579666
25: -58.5413437, 52.1953201, -58.6476593, 52.3685226, -110.9098663, 110.8429794
26: -77.7685089, 69.9219513, -77.9128647, 70.0938110, -147.8623199, 147.8347931
27: -74.4770432, 51.1829147, -74.6519318, 51.2174683, -125.6945114, 125.8348465
28: -55.3737106, 48.7586746, -55.4643745, 48.9123230, -104.2860336, 104.2230530
29: -89.3337021, 49.6997147, -89.4321747, 49.9572906, -139.2909698, 139.1318817
30: -67.7169495, 60.5408707, -67.7849884, 60.7450676, -128.4620209, 128.3258667
31: -71.8977127, 48.1686745, -72.0030136, 48.3159370, -120.2136536, 120.1716843
32: -75.1693954, 48.2472610, -75.2427979, 48.4575882, -123.6269836, 123.4900513
33: -97.6885834, 69.0671692, -97.9993286, 69.1538086, -166.8423767, 167.0664978
34: -81.3195190, 54.3290405, -81.5676422, 54.4128113, -135.7323303, 135.8966675
35: -84.0758820, 57.1555748, -84.2895355, 57.2131805, -141.2890625, 141.4450989
36: -82.1636810, 56.2845230, -82.2830353, 56.3979340, -138.5616150, 138.5675659
37: -114.6070175, 58.1342697, -114.8748703, 58.1937256, -172.8007202, 173.0091400
38: -100.3118896, 68.5343246, -100.4893646, 68.6603088, -168.9721985, 169.0236816
39: -117.3386536, 67.2530746, -117.5885468, 67.3038177, -184.6424561, 184.8416138
40: -97.1529541, 53.1961365, -97.4693527, 53.2768631, -150.4298096, 150.6654968
41: -72.5574036, 46.7620697, -72.6805267, 46.8979874, -119.4553909, 119.4425964
42: -54.4034958, 43.3686600, -54.4948540, 43.6064491, -98.0099487, 97.8635101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=186, inp2_unstable=187, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=735, inp2_unstable=734, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1121

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702898
time: 143.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4573688
time: 134.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 280.60 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 280.60
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 280.60
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 280.60
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702898
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 280.60
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4573688

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 121.74 + 2629.84 = 2751.58 seconds
