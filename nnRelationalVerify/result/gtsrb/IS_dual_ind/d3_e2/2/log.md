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
execution time: IAR + RelationalAnalysis = 2.73 + 117.08 = 119.81 seconds
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
time: 245.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
time: 566.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 812.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 812.73
Output dim: 4, lower bound: -69.5999827, upper bound: 69.6611616
IS_A2, status: Status.UNKNOWN, split count: 1, time: 812.73
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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5666756
time: 118.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5968062, upper bound: 69.6594478
time: 143.90 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6611616, upper bound: 69.5999827
time: 119.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6611616, upper bound: 69.6611616
time: 137.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 259.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 259.57
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5666756
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 259.57
Output dim: 4, lower bound: -69.5968062, upper bound: 69.6594478
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 259.57
Output dim: 4, lower bound: -69.6611616, upper bound: 69.5999827
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 259.57
Output dim: 4, lower bound: -69.6611616, upper bound: 69.6611616

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -99.8843079, 48.4684677, -99.8126602, 48.4727898, -148.3570862, 148.2811279
1: -56.4852257, 47.2781830, -56.4198227, 47.3440857, -103.8292999, 103.6980057
2: -48.4905472, 44.4175911, -48.4396667, 44.4123573, -92.9028931, 92.8572540
3: -53.5836716, 56.2901917, -53.5171890, 56.2896080, -109.8732758, 109.8073730
4: -57.3641663, 55.2840805, -57.3473129, 55.2949104, -112.6590729, 112.6313782
5: -52.0940590, 57.2839584, -52.0250473, 57.2870369, -109.3810959, 109.3090057
6: -70.0627899, 50.3417435, -70.0899124, 50.3939362, -120.4567261, 120.4316406
7: -64.7962570, 59.1574516, -64.7319183, 59.2555847, -124.0518188, 123.8893738
8: -63.6320992, 68.7627029, -63.5947075, 68.7769089, -132.4089966, 132.3574066
9: -59.1674652, 50.9124336, -59.1601257, 50.9483948, -110.1158600, 110.0725555
10: -83.5324631, 77.2600479, -83.5343399, 77.3298187, -160.8622742, 160.7943878
11: -78.6029510, 59.6022797, -78.6558838, 59.6727562, -138.2757111, 138.2581482
12: -77.3654785, 64.6213150, -77.4127197, 64.6471100, -142.0125885, 142.0340118
13: -79.9667587, 73.0594177, -79.9844208, 73.2296906, -153.1964417, 153.0438385
14: -121.7543869, 60.8298378, -121.7936401, 60.9987488, -182.7531433, 182.6234741
15: -65.8360214, 49.3779678, -65.8742828, 49.3872070, -115.2232285, 115.2522507
16: -87.7535706, 59.5806236, -87.7378845, 59.6533890, -147.4069519, 147.3184967
17: -122.8150787, 91.6125946, -122.8883362, 91.8091431, -214.6241760, 214.5009308
18: -70.3715744, 59.6791077, -70.4720612, 59.5919609, -129.9635315, 130.1511688
19: -55.3410797, 35.2319946, -55.3611946, 35.2297516, -90.5708313, 90.5931854
20: -50.4857864, 42.5618515, -50.4971504, 42.5713387, -93.0571289, 93.0590057
21: -71.3202515, 45.8627090, -71.3491211, 45.8553658, -117.1756134, 117.2118225
22: -77.8405609, 48.2031708, -77.9163513, 48.1474991, -125.9880600, 126.1195221
23: -56.6003113, 45.0612640, -56.6185684, 45.0734444, -101.6737518, 101.6798248
24: -70.5512695, 51.9037819, -70.6285248, 51.9010468, -122.4523163, 122.5323029
25: -58.5210876, 52.1471710, -58.5578194, 52.1788139, -110.6998901, 110.7049866
26: -77.7397614, 69.8190308, -77.7762756, 69.6894379, -147.4291992, 147.5953064
27: -74.4493866, 51.1517143, -74.5388336, 51.0953636, -125.5447464, 125.6905518
28: -55.3530235, 48.7047806, -55.3659515, 48.7014313, -104.0544510, 104.0707245
29: -89.3121185, 49.5918198, -89.3261871, 49.5353775, -138.8474884, 138.9179993
30: -67.6974106, 60.4714890, -67.7030945, 60.4608154, -128.1582336, 128.1745911
31: -71.8670273, 48.1164398, -71.8773956, 48.1092567, -119.9762878, 119.9938278
32: -75.1523438, 48.1737747, -75.1658936, 48.1712074, -123.3235321, 123.3396683
33: -97.6101227, 69.0442657, -97.6881714, 69.0539169, -166.6640320, 166.7324219
34: -81.2480545, 54.3088112, -81.2843170, 54.3212776, -135.5693207, 135.5931244
35: -84.0192413, 57.1422119, -84.0644531, 57.1554642, -141.1746826, 141.2066650
36: -82.1413422, 56.2323303, -82.1876678, 56.1886902, -138.3300323, 138.4199829
37: -114.5677567, 58.0891418, -114.7093811, 58.0136566, -172.5814209, 172.7985229
38: -100.2592010, 68.5045547, -100.2699890, 68.5337601, -168.7929688, 168.7745361
39: -117.2831802, 67.2435226, -117.3676682, 67.2430115, -184.5261841, 184.6111908
40: -97.0990295, 53.1797180, -97.2591248, 53.2055130, -150.3045349, 150.4388428
41: -72.5295792, 46.7195206, -72.5732117, 46.7182465, -119.2478180, 119.2927322
42: -54.3835297, 43.2932892, -54.4015121, 43.3134918, -97.6970215, 97.6947784

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1639
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4862740, upper bound: 69.5192941
time: 99.93 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5859835, upper bound: 69.5588826
time: 209.88 seconds

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

Time for backsubstitution: 2.15 seconds

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
time: 142.25 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6594479
time: 97.12 seconds

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

Time for backsubstitution: 2.17 seconds

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
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5938845
time: 107.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5968062
time: 170.27 seconds

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

Time for backsubstitution: 2.18 seconds

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

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5666755, upper bound: 69.5946254
time: 104.99 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5978846
time: 124.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 231.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.4862740, upper bound: 69.5192941
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5859835, upper bound: 69.5588826
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6561940
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5347599, upper bound: 69.6594479
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5938845
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5968062
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5666755, upper bound: 69.5946254
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 231.66
Output dim: 4, lower bound: -69.5347599, upper bound: 69.5978846

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -99.5785980, 48.3169632, -99.7556229, 48.4463043, -148.0249023, 148.0725708
1: -56.3216705, 47.1047134, -56.3895073, 47.3139877, -103.6356506, 103.4942169
2: -48.1051064, 44.3471794, -48.3720360, 44.3984833, -92.5035858, 92.7192154
3: -53.2762527, 56.1626205, -53.4604187, 56.2656174, -109.5418701, 109.6230392
4: -56.9068680, 55.1990891, -57.2683640, 55.2729950, -112.1798553, 112.4674530
5: -51.8372765, 57.1639175, -51.9779434, 57.2617912, -109.0990677, 109.1418610
6: -69.9438934, 49.9536133, -70.0556870, 50.3245201, -120.2684174, 120.0092926
7: -64.5550537, 58.9895859, -64.6889877, 59.2260437, -123.7810974, 123.6785736
8: -63.1415291, 68.6350784, -63.5095139, 68.7529755, -131.8945007, 132.1445618
9: -59.0601578, 50.5521278, -59.1400757, 50.8855972, -109.9457474, 109.6921997
10: -83.3141098, 76.2665405, -83.4953384, 77.1604919, -160.4745941, 159.7618713
11: -78.4008713, 58.7867661, -78.6197128, 59.5318909, -137.9327698, 137.4064789
12: -77.2147827, 63.6275826, -77.3857117, 64.4792480, -141.6940308, 141.0132904
13: -79.8407211, 72.5586166, -79.9588928, 73.1437073, -152.9844360, 152.5175171
14: -121.5140152, 60.0625000, -121.7501450, 60.8711967, -182.3852081, 181.8126526
15: -65.4253845, 49.2643814, -65.7966919, 49.3516846, -114.7770691, 115.0610733
16: -87.5320053, 58.9461670, -87.6909637, 59.5432777, -147.0752716, 146.6371307
17: -122.6249084, 90.4447632, -122.8542480, 91.6138382, -214.2387390, 213.2990112
18: -70.0420380, 59.4734612, -70.4140930, 59.5543823, -129.5964203, 129.8875580
19: -55.1622162, 35.1307755, -55.3293686, 35.2094574, -90.3716736, 90.4601440
20: -50.2969284, 42.4016800, -50.4633179, 42.5420990, -92.8390198, 92.8649979
21: -71.1392517, 45.6715851, -71.3166885, 45.8197479, -116.9589996, 116.9882736
22: -77.4739380, 48.0171623, -77.8509903, 48.1049881, -125.5789261, 125.8681488
23: -56.4725838, 44.9204102, -56.5941544, 45.0463867, -101.5189667, 101.5145645
24: -70.1193390, 51.8350334, -70.5538101, 51.8880157, -122.0073547, 122.3888397
25: -58.3300552, 52.0011406, -58.5229340, 52.1528931, -110.4829407, 110.5240784
26: -77.5183411, 69.4642487, -77.7372513, 69.6245117, -147.1428528, 147.2015076
27: -73.8632660, 51.0587311, -74.4392929, 51.0775070, -124.9407730, 125.4980240
28: -55.1011124, 48.6067581, -55.3228302, 48.6829720, -103.7840805, 103.9295883
29: -89.0983429, 49.3371582, -89.2881927, 49.4897995, -138.5881348, 138.6253510
30: -67.5469971, 60.2502403, -67.6702271, 60.4188309, -127.9658127, 127.9204712
31: -71.5860291, 48.0011139, -71.8264389, 48.0868607, -119.6728821, 119.8275528
32: -74.9932709, 47.8617630, -75.1372223, 48.1156235, -123.1088943, 122.9989777
33: -97.0272827, 68.9131622, -97.5876923, 69.0307159, -166.0579987, 166.5008545
34: -80.9168854, 54.1888275, -81.2258453, 54.2995834, -135.2164612, 135.4146729
35: -83.4939270, 57.0411072, -83.9735641, 57.1376419, -140.6315613, 141.0146790
36: -81.8175354, 56.1556435, -82.1304092, 56.1701355, -137.9876709, 138.2860413
37: -114.0855560, 57.9970398, -114.6245117, 57.9922409, -172.0777893, 172.6215363
38: -99.8896790, 68.3253479, -100.2016296, 68.5018616, -168.3915405, 168.5269775
39: -116.7845154, 67.1682663, -117.2744522, 67.2284470, -184.0129547, 184.4427185
40: -96.5696487, 53.1083946, -97.1689987, 53.1918793, -149.7615204, 150.2773895
41: -72.3190536, 46.5637627, -72.5348740, 46.6884995, -119.0075531, 119.0986328
42: -54.2617531, 42.6929092, -54.3787308, 43.2105942, -97.4723511, 97.0716248

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 533
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
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1247
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
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1720
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
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1269
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
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1184
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
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4862740, upper bound: 69.4573688
time: 128.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4862740, upper bound: 69.5192941
time: 115.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -99.8800507, 48.4665413, -99.8126602, 48.4727898, -148.3528442, 148.2792053
1: -56.4825974, 47.2755089, -56.4198227, 47.3440857, -103.8266754, 103.6953278
2: -48.4880447, 44.4165649, -48.4396667, 44.4123573, -92.9004059, 92.8562317
3: -53.5795288, 56.2886772, -53.5171890, 56.2896080, -109.8691406, 109.8058624
4: -57.3572845, 55.2825012, -57.3473129, 55.2949104, -112.6521912, 112.6298141
5: -52.0902023, 57.2825165, -52.0250473, 57.2870369, -109.3772430, 109.3075562
6: -70.0606461, 50.3325424, -70.0899124, 50.3939362, -120.4545746, 120.4224472
7: -64.7935791, 59.1525650, -64.7319183, 59.2555847, -124.0491638, 123.8844833
8: -63.6281357, 68.7611237, -63.5947075, 68.7769089, -132.4050446, 132.3558350
9: -59.1659088, 50.9094162, -59.1601257, 50.9483948, -110.1143036, 110.0695419
10: -83.5302734, 77.2529907, -83.5343399, 77.3298187, -160.8600769, 160.7873230
11: -78.6010437, 59.5979309, -78.6558838, 59.6727562, -138.2738037, 138.2537994
12: -77.3637314, 64.6146240, -77.4127197, 64.6471100, -142.0108337, 142.0273438
13: -79.9637146, 73.0558319, -79.9844208, 73.2296906, -153.1933899, 153.0402374
14: -121.7508240, 60.8251266, -121.7936401, 60.9987488, -182.7495728, 182.6187592
15: -65.8279343, 49.3760910, -65.8742828, 49.3872070, -115.2151413, 115.2503738
16: -87.7505264, 59.5752335, -87.7378845, 59.6533890, -147.4039154, 147.3131104
17: -122.8125763, 91.6054688, -122.8883362, 91.8091431, -214.6217041, 214.4938049
18: -70.3662109, 59.6748276, -70.4720612, 59.5919609, -129.9581604, 130.1468811
19: -55.3375854, 35.2297554, -55.3611946, 35.2297516, -90.5673370, 90.5909500
20: -50.4838028, 42.5601807, -50.4971504, 42.5713387, -93.0551453, 93.0573273
21: -71.3183441, 45.8597145, -71.3491211, 45.8553658, -117.1736984, 117.2088318
22: -77.8317871, 48.2006302, -77.9163513, 48.1474991, -125.9792862, 126.1169739
23: -56.5988693, 45.0576935, -56.6185684, 45.0734444, -101.6723022, 101.6762619
24: -70.5475388, 51.9025726, -70.6285248, 51.9010468, -122.4485855, 122.5310898
25: -58.5172081, 52.1451988, -58.5578194, 52.1788139, -110.6960144, 110.7030182
26: -77.7347717, 69.8144073, -77.7762756, 69.6894379, -147.4242096, 147.5906677
27: -74.4454346, 51.1497803, -74.5388336, 51.0953636, -125.5408020, 125.6886139
28: -55.3510818, 48.7027550, -55.3659515, 48.7014313, -104.0524979, 104.0686951
29: -89.3067932, 49.5889664, -89.3261871, 49.5353775, -138.8421631, 138.9151306
30: -67.6956024, 60.4633713, -67.7030945, 60.4608154, -128.1564026, 128.1664734
31: -71.8636475, 48.1139297, -71.8773956, 48.1092567, -119.9728851, 119.9913101
32: -75.1507568, 48.1703606, -75.1658936, 48.1712074, -123.3219299, 123.3362579
33: -97.6053925, 69.0429688, -97.6881714, 69.0539169, -166.6593018, 166.7311401
34: -81.2449036, 54.3058510, -81.2843170, 54.3212776, -135.5661774, 135.5901642
35: -84.0147705, 57.1411209, -84.0644531, 57.1554642, -141.1702271, 141.2055664
36: -82.1383209, 56.2309952, -82.1876678, 56.1886902, -138.3270111, 138.4186401
37: -114.5621948, 58.0878601, -114.7093811, 58.0136566, -172.5758362, 172.7972412
38: -100.2545853, 68.5001984, -100.2699890, 68.5337601, -168.7883453, 168.7701874
39: -117.2774658, 67.2424774, -117.3676682, 67.2430115, -184.5204620, 184.6101379
40: -97.0952606, 53.1780624, -97.2591248, 53.2055130, -150.3007507, 150.4371948
41: -72.5276184, 46.7146645, -72.5732117, 46.7182465, -119.2458649, 119.2878723
42: -54.3819733, 43.2877846, -54.4015121, 43.3134918, -97.6954651, 97.6892853

Time for backsubstitution: 2.17 seconds

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
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1247
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
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1242
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
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1269
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
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5471515, upper bound: 69.4573688
time: 111.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5471515, upper bound: 69.4573688
time: 109.35 seconds

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

Time for backsubstitution: 2.15 seconds

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
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
time: 114.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
time: 117.78 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702898
time: 143.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4573688
time: 133.04 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -99.8824158, 48.5000648, -99.8843079, 48.4684677, -148.3508911, 148.3843689
1: -56.4502945, 47.3757782, -56.4852257, 47.2781830, -103.7284775, 103.8610077
2: -48.5021477, 44.4325752, -48.4905472, 44.4175911, -92.9197311, 92.9231110
3: -53.5893402, 56.3170471, -53.5836716, 56.2901917, -109.8795242, 109.9007111
4: -57.4439545, 55.3144455, -57.3641663, 55.2840805, -112.7280197, 112.6786118
5: -52.0824966, 57.3192215, -52.0940590, 57.2839584, -109.3664474, 109.4132843
6: -70.1204376, 50.4940300, -70.0627899, 50.3417435, -120.4621811, 120.5568085
7: -64.7698517, 59.3020477, -64.7962570, 59.1574516, -123.9272995, 124.0982971
8: -63.6779251, 68.8061676, -63.6320992, 68.7627029, -132.4406281, 132.4382629
9: -59.2199478, 50.9868927, -59.1674652, 50.9124336, -110.1323853, 110.1543579
10: -83.5816193, 77.4334106, -83.5324631, 77.2600479, -160.8416443, 160.9658813
11: -78.6992035, 59.8613358, -78.6029510, 59.6022797, -138.3014832, 138.4642792
12: -77.4480133, 64.8210068, -77.3654785, 64.6213150, -142.0693207, 142.1864777
13: -80.0153503, 73.3495636, -79.9667587, 73.0594177, -153.0747681, 153.3163147
14: -121.8400803, 61.1576424, -121.7543869, 60.8298378, -182.6698914, 182.9120331
15: -66.0031891, 49.4218292, -65.8360214, 49.3779678, -115.3811493, 115.2578506
16: -87.7932358, 59.7523651, -87.7535706, 59.5806236, -147.3738403, 147.5059357
17: -122.9374084, 92.0935364, -122.8150787, 91.6125946, -214.5499878, 214.9085846
18: -70.5388794, 59.6256676, -70.3715744, 59.6791077, -130.2179871, 129.9972382
19: -55.3957901, 35.3137512, -55.3410797, 35.2319946, -90.6277847, 90.6548309
20: -50.5270538, 42.6403885, -50.4857864, 42.5618515, -93.0888977, 93.1261749
21: -71.3864670, 45.9584122, -71.3202515, 45.8627090, -117.2491684, 117.2786636
22: -77.9680405, 48.2084656, -77.8405609, 48.2031708, -126.1712112, 126.0490265
23: -56.6460228, 45.1325531, -56.6003113, 45.0612640, -101.7072830, 101.7328568
24: -70.6892929, 51.9160233, -70.5512695, 51.9037819, -122.5930786, 122.4672852
25: -58.5956039, 52.2442780, -58.5210876, 52.1471710, -110.7427750, 110.7653656
26: -77.8184738, 69.7576675, -77.7397614, 69.8190308, -147.6375122, 147.4974365
27: -74.6022034, 51.1169243, -74.4493866, 51.1517143, -125.7539215, 125.5663071
28: -55.3929443, 48.7639618, -55.3530235, 48.7047806, -104.0977020, 104.1169891
29: -89.3582840, 49.6344376, -89.3121185, 49.5918198, -138.9501038, 138.9465332
30: -67.7372055, 60.5249252, -67.6974106, 60.4714890, -128.2086792, 128.2223206
31: -71.9238281, 48.1650467, -71.8670273, 48.1164398, -120.0402603, 120.0320740
32: -75.1919174, 48.2534332, -75.1523438, 48.1737747, -123.3656845, 123.4057770
33: -97.8035660, 69.0814743, -97.6101227, 69.0442657, -166.8478088, 166.6915894
34: -81.3769455, 54.3472977, -81.2480545, 54.3088112, -135.6857605, 135.5953369
35: -84.1442566, 57.1739655, -84.0192413, 57.1422119, -141.2864532, 141.1932068
36: -82.2267838, 56.2310562, -82.1413422, 56.2323303, -138.4591064, 138.3724060
37: -114.8076324, 58.0367050, -114.5677567, 58.0891418, -172.8967743, 172.6044617
38: -100.3249054, 68.5808868, -100.2592010, 68.5045547, -168.8294678, 168.8400879
39: -117.4612808, 67.2561493, -117.2831802, 67.2435226, -184.7048035, 184.5393372
40: -97.3770599, 53.2261734, -97.0990295, 53.1797180, -150.5567780, 150.3251953
41: -72.6154556, 46.7723656, -72.5295792, 46.7195206, -119.3349686, 119.3019409
42: -54.4283180, 43.4089394, -54.3835297, 43.2932892, -97.7216034, 97.7924500

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1247
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
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1242
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
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1269
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
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1184
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
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1639
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4892908, upper bound: 69.4862740
time: 201.26 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4892908, upper bound: 69.5859835
time: 143.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -100.2075043, 48.5755615, -99.9720459, 48.4851074, -148.6926117, 148.5476074
1: -56.6460266, 47.4193687, -56.5383034, 47.2901535, -103.9361725, 103.9576721
2: -48.7822342, 44.4997559, -48.5665932, 44.4318008, -93.2140350, 93.0663452
3: -53.9216728, 56.4062119, -53.6739273, 56.3102913, -110.2319489, 110.0801392
4: -57.7948265, 55.3902359, -57.4597931, 55.2988892, -113.0937195, 112.8500290
5: -52.3742676, 57.4115791, -52.1733780, 57.3041458, -109.6784134, 109.5849609
6: -70.2138519, 50.7680969, -70.0861893, 50.4166412, -120.6304779, 120.8542786
7: -64.9595032, 59.3613701, -64.8497620, 59.1724930, -124.1319962, 124.2111359
8: -64.0075531, 68.8876038, -63.7219849, 68.7808380, -132.7883911, 132.6095886
9: -59.4474449, 51.0888863, -59.2283745, 50.9391785, -110.3866272, 110.3172455
10: -83.7556686, 77.6950150, -83.5794830, 77.3305359, -161.0861969, 161.2745056
11: -78.8348389, 60.4134750, -78.6325150, 59.7525635, -138.5874023, 139.0459900
12: -77.5677795, 65.3952179, -77.3877106, 64.7767181, -142.3444977, 142.7829285
13: -80.1085205, 73.5530014, -79.9936829, 73.1153107, -153.2238312, 153.5466919
14: -122.0027390, 61.5038300, -121.7941589, 60.9223328, -182.9250793, 183.2979889
15: -66.4219437, 49.4657555, -65.9452057, 49.4036674, -115.8255920, 115.4109650
16: -87.9740219, 60.0088005, -87.8031311, 59.6514816, -147.6255035, 147.8119354
17: -123.1269379, 92.8332977, -122.8538666, 91.8122711, -214.9392090, 215.6871643
18: -70.6837387, 59.8447456, -70.4102325, 59.7390633, -130.4227905, 130.2549744
19: -55.5155067, 35.5901871, -55.3677597, 35.3070526, -90.8225555, 90.9579468
20: -50.6267662, 42.8732452, -50.5101318, 42.6247025, -93.2514572, 93.3833771
21: -71.5089264, 46.3299103, -71.3475266, 45.9627457, -117.4716644, 117.6774292
22: -78.0755539, 48.5062637, -77.8689270, 48.2847748, -126.3603287, 126.3751907
23: -56.7351608, 45.3234253, -56.6209869, 45.1135712, -101.8487320, 101.9443970
24: -70.7998810, 51.9783592, -70.5814209, 51.9221916, -122.7220688, 122.5597839
25: -58.6856003, 52.4345169, -58.5443649, 52.1989365, -110.8845367, 110.9788818
26: -77.9549179, 70.1628265, -77.7713547, 69.9282532, -147.8831482, 147.9341431
27: -74.7153625, 51.2393341, -74.4797821, 51.1869164, -125.9022827, 125.7191086
28: -55.4913979, 48.9748917, -55.3755417, 48.7620010, -104.2534027, 104.3504333
29: -89.4642792, 50.0569763, -89.3361664, 49.7068253, -139.1710968, 139.3931427
30: -67.8194427, 60.8093338, -67.7197952, 60.5500717, -128.3695068, 128.5291290
31: -72.0495224, 48.3746605, -71.9006500, 48.1720810, -120.2215958, 120.2753143
32: -75.2689285, 48.5399170, -75.1715393, 48.2525291, -123.5214539, 123.7114410
33: -98.1150513, 69.1814575, -97.6944275, 69.0692139, -167.1842651, 166.8758850
34: -81.6604996, 54.4387665, -81.3240280, 54.3310318, -135.9915161, 135.7627869
35: -84.3696136, 57.2317810, -84.0802307, 57.1569366, -141.5265503, 141.3120117
36: -82.3221588, 56.4407578, -82.1658478, 56.2879753, -138.6101379, 138.6065979
37: -114.9740753, 58.2172852, -114.6117401, 58.1408768, -173.1149597, 172.8290253
38: -100.5451355, 68.7077637, -100.3190918, 68.5386124, -169.0837402, 169.0268555
39: -117.6828003, 67.3168488, -117.3436661, 67.2601929, -184.9429932, 184.6605072
40: -97.5883102, 53.2974701, -97.1583176, 53.1989746, -150.7872772, 150.4557800
41: -72.7228165, 46.9537773, -72.5600281, 46.7714386, -119.4942551, 119.5138092
42: -54.5216751, 43.7017403, -54.4054337, 43.3735962, -97.8952713, 98.1071777

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1247
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
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1242
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
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1269
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
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1184
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
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1639
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6197169, upper bound: 69.5347599
time: 186.20 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.6218561, upper bound: 69.5968062
time: 118.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -99.8824158, 48.5000648, -100.1256790, 48.5602036, -148.4426270, 148.6257324
1: -56.4502945, 47.3757782, -56.5966911, 47.4083405, -103.8586349, 103.9724731
2: -48.5021477, 44.4325752, -48.7111206, 44.4869881, -92.9891357, 93.1436920
3: -53.5893402, 56.3170471, -53.8370018, 56.3885422, -109.9778824, 110.1540375
4: -57.4439545, 55.3144455, -57.7055511, 55.3768654, -112.8208160, 113.0199890
5: -52.0824966, 57.3192215, -52.3001709, 57.3937492, -109.4762421, 109.6193848
6: -70.1204376, 50.4940300, -70.1928864, 50.7000885, -120.8205261, 120.6869049
7: -64.7698517, 59.3020477, -64.9112778, 59.3473816, -124.1172180, 124.2133255
8: -63.6779251, 68.8061676, -63.9235687, 68.8713684, -132.5492859, 132.7297363
9: -59.2199478, 50.9868927, -59.3889008, 51.0655632, -110.2855072, 110.3757935
10: -83.5816193, 77.4334106, -83.7145233, 77.6305084, -161.2121277, 161.1479340
11: -78.6992035, 59.8613358, -78.8082428, 60.2720566, -138.9712524, 138.6695862
12: -77.4480133, 64.8210068, -77.5476685, 65.2492981, -142.6973114, 142.3686523
13: -80.0153503, 73.3495636, -80.0870667, 73.5036011, -153.5189514, 153.4366302
14: -121.8400803, 61.1576424, -121.9668655, 61.4161758, -183.2562561, 183.1245117
15: -66.0031891, 49.4218292, -66.3148193, 49.4426804, -115.4458618, 115.7366486
16: -87.7932358, 59.7523651, -87.9301910, 59.9548988, -147.7481232, 147.6825562
17: -122.9374084, 92.0935364, -123.0913391, 92.6445618, -215.5819550, 215.1848602
18: -70.5388794, 59.6256676, -70.6517944, 59.7886353, -130.3275146, 130.2774506
19: -55.3957901, 35.3137512, -55.4915390, 35.5195312, -90.9153214, 90.8052902
20: -50.5270538, 42.6403885, -50.6043053, 42.8140526, -93.3410950, 93.2446899
21: -71.3864670, 45.9584122, -71.4842682, 46.2353516, -117.6218185, 117.4426804
22: -77.9680405, 48.2084656, -78.0497818, 48.4295578, -126.3975983, 126.2582474
23: -56.6460228, 45.1325531, -56.7163429, 45.2744904, -101.9205170, 101.8488922
24: -70.6892929, 51.9160233, -70.7746201, 51.9638405, -122.6531372, 122.6906433
25: -58.5956039, 52.2442780, -58.6652756, 52.3858795, -110.9814835, 110.9095535
26: -77.8184738, 69.7576675, -77.9258881, 70.0598602, -147.8783264, 147.6835480
27: -74.6022034, 51.1169243, -74.6872253, 51.2090836, -125.8112869, 125.8041382
28: -55.3929443, 48.7639618, -55.4708595, 48.9211807, -104.3141174, 104.2348175
29: -89.3582840, 49.6344376, -89.4427032, 49.9489441, -139.3072205, 139.0771332
30: -67.7372055, 60.5249252, -67.7994308, 60.7402039, -128.4774017, 128.3243408
31: -71.9238281, 48.1650467, -72.0190887, 48.3198624, -120.2436752, 120.1841354
32: -75.1919174, 48.2534332, -75.2517548, 48.4670525, -123.6589508, 123.5051880
33: -97.8035660, 69.0814743, -98.0364685, 69.1583710, -166.9619141, 167.1179504
34: -81.3769455, 54.3472977, -81.5890503, 54.4186249, -135.7955627, 135.9363403
35: -84.1442566, 57.1739655, -84.3129578, 57.2183533, -141.3625946, 141.4869232
36: -82.2267838, 56.2310562, -82.2997742, 56.3882828, -138.6150665, 138.5308228
37: -114.8076324, 58.0367050, -114.9339142, 58.1724358, -172.9800720, 172.9706116
38: -100.3249054, 68.5808868, -100.4918976, 68.6777115, -169.0026245, 169.0727844
39: -117.4612808, 67.2561493, -117.6271210, 67.3072815, -184.7685547, 184.8832703
40: -97.3770599, 53.2261734, -97.5334549, 53.2813416, -150.6584015, 150.7596130
41: -72.6154556, 46.7723656, -72.6950150, 46.9110107, -119.5264664, 119.4673767
42: -54.4283180, 43.4089394, -54.5017548, 43.6270638, -98.0553818, 97.9106827

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4927358
time: 134.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5588825, upper bound: 69.5866541
time: 124.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -100.2075043, 48.5755615, -100.2133789, 48.5769882, -148.7844849, 148.7889404
1: -56.6460266, 47.4193687, -56.6496010, 47.4203453, -104.0663757, 104.0689697
2: -48.7822342, 44.4997559, -48.7867737, 44.5010529, -93.2832870, 93.2865219
3: -53.9216728, 56.4062119, -53.9270096, 56.4083099, -110.3299866, 110.3332214
4: -57.7948265, 55.3902359, -57.8006477, 55.3915558, -113.1863861, 113.1908798
5: -52.3742676, 57.4115791, -52.3790817, 57.4138107, -109.7880554, 109.7906494
6: -70.2138519, 50.7680969, -70.2161560, 50.7749023, -120.9887390, 120.9842529
7: -64.9595032, 59.3613701, -64.9640198, 59.3626785, -124.3221817, 124.3253937
8: -64.0075531, 68.8876038, -64.0129700, 68.8893433, -132.8968964, 132.9005737
9: -59.4474449, 51.0888863, -59.4515572, 51.0926132, -110.5400543, 110.5404205
10: -83.7556686, 77.6950150, -83.7617493, 77.7007599, -161.4564209, 161.4567566
11: -78.8348389, 60.4134750, -78.8379364, 60.4215965, -139.2564392, 139.2514038
12: -77.5677795, 65.3952179, -77.5697021, 65.4040375, -142.9718170, 142.9649200
13: -80.1085205, 73.5530014, -80.1138000, 73.5606079, -153.6691284, 153.6668091
14: -122.0027390, 61.5038300, -122.0064392, 61.5088577, -183.5115967, 183.5102692
15: -66.4219437, 49.4657555, -66.4320068, 49.4682465, -115.8901749, 115.8977661
16: -87.9740219, 60.0088005, -87.9804077, 60.0269012, -148.0009155, 147.9892120
17: -123.1269379, 92.8332977, -123.1297607, 92.8440170, -215.9709473, 215.9630432
18: -70.6837387, 59.8447456, -70.6920471, 59.8486748, -130.5324097, 130.5367889
19: -55.5155067, 35.5901871, -55.5178604, 35.5943375, -91.1098480, 91.1080475
20: -50.6267662, 42.8732452, -50.6286926, 42.8769264, -93.5036926, 93.5019379
21: -71.5089264, 46.3299103, -71.5113525, 46.3352242, -117.8441391, 117.8412628
22: -78.0755539, 48.5062637, -78.0781174, 48.5118103, -126.5873642, 126.5843811
23: -56.7351608, 45.3234253, -56.7367783, 45.3266869, -102.0618439, 102.0602036
24: -70.7998810, 51.9783592, -70.8049088, 51.9816895, -122.7815704, 122.7832642
25: -58.6856003, 52.4345169, -58.6886520, 52.4380302, -111.1236267, 111.1231689
26: -77.9549179, 70.1628265, -77.9576569, 70.1691742, -148.1240845, 148.1204834
27: -74.7153625, 51.2393341, -74.7180252, 51.2440720, -125.9594269, 125.9573593
28: -55.4913979, 48.9748917, -55.4932022, 48.9783173, -104.4697113, 104.4680939
29: -89.4642792, 50.0569763, -89.4667206, 50.0640640, -139.5283508, 139.5236969
30: -67.8194427, 60.8093338, -67.8220673, 60.8186874, -128.6381226, 128.6314087
31: -72.0495224, 48.3746605, -72.0523834, 48.3770828, -120.4266052, 120.4270325
32: -75.2689285, 48.5399170, -75.2709808, 48.5453568, -123.8142776, 123.8108826
33: -98.1150513, 69.1814575, -98.1208801, 69.1833649, -167.2984161, 167.3023224
34: -81.6604996, 54.4387665, -81.6650772, 54.4407387, -136.1012268, 136.1038361
35: -84.3696136, 57.2317810, -84.3739929, 57.2330589, -141.6026764, 141.6057739
36: -82.3221588, 56.4407578, -82.3243484, 56.4442749, -138.7664185, 138.7651062
37: -114.9740753, 58.2172852, -114.9787750, 58.2241440, -173.1982117, 173.1960602
38: -100.5451355, 68.7077637, -100.5520401, 68.7120667, -169.2572021, 169.2597961
39: -117.6828003, 67.3168488, -117.6877213, 67.3240509, -185.0068512, 185.0045776
40: -97.5883102, 53.2974701, -97.5934296, 53.3004608, -150.8887634, 150.8908997
41: -72.7228165, 46.9537773, -72.7253723, 46.9633675, -119.6861877, 119.6791382
42: -54.5216751, 43.7017403, -54.5235481, 43.7069016, -98.2285690, 98.2252808

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5357092
time: 107.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.5968062, upper bound: 69.5978847
time: 116.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 226.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4862740, upper bound: 69.4573688
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4862740, upper bound: 69.5192941
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.5471515, upper bound: 69.4573688
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.5471515, upper bound: 69.4573688
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4833982, upper bound: 69.5523038
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702898
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4573688
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4892908, upper bound: 69.4862740
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4892908, upper bound: 69.5859835
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.6197169, upper bound: 69.5347599
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.6218561, upper bound: 69.5968062
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4927358
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.5588825, upper bound: 69.5866541
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5357092
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 226.25
Output dim: 4, lower bound: -69.5968062, upper bound: 69.5978847

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -99.5785980, 48.3169632, -99.5050430, 48.3204346, -147.8990173, 147.8220062
1: -56.3216705, 47.1047134, -56.2561531, 47.1704292, -103.4920959, 103.3608704
2: -48.1051064, 44.3471794, -48.0534935, 44.3419991, -92.4470978, 92.4006729
3: -53.2762527, 56.1626205, -53.2089539, 56.1622620, -109.4385147, 109.3715744
4: -56.9068680, 55.1990891, -56.8895950, 55.2103500, -112.1172180, 112.0886765
5: -51.8372765, 57.1639175, -51.7674026, 57.1669922, -109.0042725, 108.9313126
6: -69.9438934, 49.9536133, -69.9705200, 50.0056839, -119.9495773, 119.9241257
7: -64.5550537, 58.9895859, -64.4916077, 59.0869904, -123.6420441, 123.4811935
8: -63.1415291, 68.6350784, -63.1033897, 68.6500549, -131.7915802, 131.7384644
9: -59.0601578, 50.5521278, -59.0530434, 50.5876236, -109.6477814, 109.6051712
10: -83.3141098, 76.2665405, -83.3163605, 76.3353119, -159.6494141, 159.5829010
11: -78.4008713, 58.7867661, -78.4558411, 58.8566895, -137.2575684, 137.2425995
12: -77.2147827, 63.6275826, -77.2626343, 63.6525764, -140.8673553, 140.8902130
13: -79.8407211, 72.5586166, -79.8593216, 72.7272034, -152.5679321, 152.4179382
14: -121.5140152, 60.0625000, -121.5538635, 60.2309952, -181.7450104, 181.6163635
15: -65.4253845, 49.2643814, -65.4629517, 49.2740631, -114.6994476, 114.7273331
16: -87.5320053, 58.9461670, -87.5159836, 59.0184517, -146.5504456, 146.4621582
17: -122.6249084, 90.4447632, -122.6986008, 90.6406021, -213.2655029, 213.1433563
18: -70.0420380, 59.4734612, -70.1410904, 59.3860245, -129.4280701, 129.6145325
19: -55.1622162, 35.1307755, -55.1831207, 35.1282463, -90.2904663, 90.3138962
20: -50.2969284, 42.4016800, -50.3081932, 42.4107399, -92.7076569, 92.7098694
21: -71.1392517, 45.6715851, -71.1681976, 45.6635437, -116.8027954, 116.8397827
22: -77.4739380, 48.0171623, -77.5496216, 47.9614296, -125.4353561, 125.5667877
23: -56.4725838, 44.9204102, -56.4921837, 44.9324265, -101.4050140, 101.4125977
24: -70.1193390, 51.8350334, -70.1962585, 51.8323669, -121.9517059, 122.0312805
25: -58.3300552, 52.0011406, -58.3666573, 52.0321503, -110.3621902, 110.3677979
26: -77.5183411, 69.4642487, -77.5539398, 69.3323364, -146.8506775, 147.0181885
27: -73.8632660, 51.0587311, -73.9499207, 51.0027199, -124.8659821, 125.0086517
28: -55.1011124, 48.6067581, -55.1145325, 48.6033134, -103.7044220, 103.7212906
29: -89.0983429, 49.3371582, -89.1124573, 49.2797546, -138.3780975, 138.4496155
30: -67.5469971, 60.2502403, -67.5522537, 60.2392807, -127.7862701, 127.8024902
31: -71.5860291, 48.0011139, -71.5968170, 47.9929199, -119.5789490, 119.5979309
32: -74.9932709, 47.8617630, -75.0070343, 47.8587952, -122.8520660, 122.8687897
33: -97.0272827, 68.9131622, -97.1052399, 68.9231720, -165.9504547, 166.0184021
34: -80.9168854, 54.1888275, -80.9523926, 54.2012787, -135.1181488, 135.1412048
35: -83.4939270, 57.0411072, -83.5387497, 57.0546494, -140.5485840, 140.5798645
36: -81.8175354, 56.1556435, -81.8634949, 56.1118698, -137.9294128, 138.0191345
37: -114.0855560, 57.9970398, -114.2252502, 57.9223328, -172.0078735, 172.2222900
38: -99.8896790, 68.3253479, -99.8996582, 68.3540039, -168.2436829, 168.2250061
39: -116.7845154, 67.1682663, -116.8689423, 67.1679916, -183.9525146, 184.0372009
40: -96.5696487, 53.1083946, -96.7294617, 53.1348114, -149.7044525, 149.8378601
41: -72.3190536, 46.5637627, -72.3625641, 46.5618057, -118.8808594, 118.9263306
42: -54.2617531, 42.6929092, -54.2802391, 42.7130051, -96.9747543, 96.9731445

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1561
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
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1702
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 155.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 107.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -99.5785980, 48.3169632, -99.8083878, 48.4708862, -148.0494690, 148.1253357
1: -56.3216705, 47.1047134, -56.4171753, 47.3414001, -103.6630554, 103.5218811
2: -48.1051064, 44.3471794, -48.4371681, 44.4113235, -92.5164337, 92.7843475
3: -53.2762527, 56.1626205, -53.5130424, 56.2880859, -109.5643234, 109.6756592
4: -56.9068680, 55.1990891, -57.3404083, 55.2933311, -112.2001953, 112.5394897
5: -51.8372765, 57.1639175, -52.0211639, 57.2855911, -109.1228638, 109.1850586
6: -69.9438934, 49.9536133, -70.0877686, 50.3846397, -120.3285370, 120.0413818
7: -64.5550537, 58.9895859, -64.7292252, 59.2506981, -123.8057556, 123.7188110
8: -63.1415291, 68.6350784, -63.5907135, 68.7753448, -131.9168701, 132.2257843
9: -59.0601578, 50.5521278, -59.1585693, 50.9453964, -110.0055542, 109.7106934
10: -83.3141098, 76.2665405, -83.5321350, 77.3227081, -160.6368103, 159.7986755
11: -78.4008713, 58.7867661, -78.6539917, 59.6684265, -138.0693054, 137.4407654
12: -77.2147827, 63.6275826, -77.4109650, 64.6404114, -141.8551941, 141.0385437
13: -79.8407211, 72.5586166, -79.9813843, 73.2260895, -153.0668030, 152.5400085
14: -121.5140152, 60.0625000, -121.7901154, 60.9940796, -182.5080872, 181.8526154
15: -65.4253845, 49.2643814, -65.8662109, 49.3853340, -114.8107147, 115.1305923
16: -87.5320053, 58.9461670, -87.7348022, 59.6479912, -147.1799927, 146.6809692
17: -122.6249084, 90.4447632, -122.8858795, 91.8020020, -214.4269104, 213.3306427
18: -70.0420380, 59.4734612, -70.4666901, 59.5876732, -129.6297150, 129.9401550
19: -55.1622162, 35.1307755, -55.3576927, 35.2275124, -90.3897247, 90.4884644
20: -50.2969284, 42.4016800, -50.4951706, 42.5696487, -92.8665695, 92.8968506
21: -71.1392517, 45.6715851, -71.3471985, 45.8523712, -116.9916229, 117.0187836
22: -77.4739380, 48.0171623, -77.9075394, 48.1449509, -125.6188812, 125.9246979
23: -56.4725838, 44.9204102, -56.6171379, 45.0698700, -101.5424500, 101.5375519
24: -70.1193390, 51.8350334, -70.6248016, 51.8998260, -122.0191650, 122.4598312
25: -58.3300552, 52.0011406, -58.5539398, 52.1767960, -110.5068512, 110.5550766
26: -77.5183411, 69.4642487, -77.7713013, 69.6847763, -147.2031250, 147.2355499
27: -73.8632660, 51.0587311, -74.5348587, 51.0934143, -124.9566803, 125.5935898
28: -55.1011124, 48.6067581, -55.3639984, 48.6994133, -103.8005142, 103.9707489
29: -89.0983429, 49.3371582, -89.3207703, 49.5324974, -138.6307983, 138.6579285
30: -67.5469971, 60.2502403, -67.7012863, 60.4526558, -127.9996490, 127.9515228
31: -71.5860291, 48.0011139, -71.8739777, 48.1067429, -119.6927719, 119.8750916
32: -74.9932709, 47.8617630, -75.1643066, 48.1677856, -123.1610565, 123.0260620
33: -97.0272827, 68.9131622, -97.6834106, 69.0526276, -166.0799103, 166.5965729
34: -80.9168854, 54.1888275, -81.2811584, 54.3182373, -135.2351227, 135.4699860
35: -83.4939270, 57.0411072, -84.0599823, 57.1543503, -140.6482544, 141.1010895
36: -81.8175354, 56.1556435, -82.1846466, 56.1873398, -138.0048523, 138.3402863
37: -114.0855560, 57.9970398, -114.7038345, 58.0123901, -172.0979309, 172.7008667
38: -99.8896790, 68.3253479, -100.2653656, 68.5293961, -168.4190521, 168.5907135
39: -116.7845154, 67.1682663, -117.3619385, 67.2419739, -184.0264893, 184.5302124
40: -96.5696487, 53.1083946, -97.2553635, 53.2038460, -149.7734985, 150.3637543
41: -72.3190536, 46.5637627, -72.5712585, 46.7134285, -119.0324860, 119.1350250
42: -54.2617531, 42.6929092, -54.3999481, 43.3079720, -97.5697174, 97.0928574

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1561
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
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1702
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.5192941
time: 179.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.5192941
time: 103.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -99.8800507, 48.4665413, -99.5050430, 48.3204346, -148.2004852, 147.9715881
1: -56.4825974, 47.2755089, -56.2561531, 47.1704292, -103.6530304, 103.5316620
2: -48.4880447, 44.4165649, -48.0534935, 44.3419991, -92.8300323, 92.4700623
3: -53.5795288, 56.2886772, -53.2089539, 56.1622620, -109.7417908, 109.4976273
4: -57.3572845, 55.2825012, -56.8895950, 55.2103500, -112.5676193, 112.1720963
5: -52.0902023, 57.2825165, -51.7674026, 57.1669922, -109.2571945, 109.0499115
6: -70.0606461, 50.3325424, -69.9705200, 50.0056839, -120.0663223, 120.3030624
7: -64.7935791, 59.1525650, -64.4916077, 59.0869904, -123.8805695, 123.6441727
8: -63.6281357, 68.7611237, -63.1033897, 68.6500549, -132.2781982, 131.8645172
9: -59.1659088, 50.9094162, -59.0530434, 50.5876236, -109.7535324, 109.9624634
10: -83.5302734, 77.2529907, -83.3163605, 76.3353119, -159.8655701, 160.5693512
11: -78.6010437, 59.5979309, -78.4558411, 58.8566895, -137.4577332, 138.0537720
12: -77.3637314, 64.6146240, -77.2626343, 63.6525764, -141.0163116, 141.8772583
13: -79.9637146, 73.0558319, -79.8593216, 72.7272034, -152.6909180, 152.9151306
14: -121.7508240, 60.8251266, -121.5538635, 60.2309952, -181.9818115, 182.3789978
15: -65.8279343, 49.3760910, -65.4629517, 49.2740631, -115.1019897, 114.8390427
16: -87.7505264, 59.5752335, -87.5159836, 59.0184517, -146.7689819, 147.0912170
17: -122.8125763, 91.6054688, -122.6986008, 90.6406021, -213.4531860, 214.3040771
18: -70.3662109, 59.6748276, -70.1410904, 59.3860245, -129.7522278, 129.8159180
19: -55.3375854, 35.2297554, -55.1831207, 35.1282463, -90.4658356, 90.4128723
20: -50.4838028, 42.5601807, -50.3081932, 42.4107399, -92.8945465, 92.8683701
21: -71.3183441, 45.8597145, -71.1681976, 45.6635437, -116.9818878, 117.0279083
22: -77.8317871, 48.2006302, -77.5496216, 47.9614296, -125.7932053, 125.7502518
23: -56.5988693, 45.0576935, -56.4921837, 44.9324265, -101.5312958, 101.5498810
24: -70.5475388, 51.9025726, -70.1962585, 51.8323669, -122.3799057, 122.0988159
25: -58.5172081, 52.1451988, -58.3666573, 52.0321503, -110.5493393, 110.5118561
26: -77.7347717, 69.8144073, -77.5539398, 69.3323364, -147.0671082, 147.3683472
27: -74.4454346, 51.1497803, -73.9499207, 51.0027199, -125.4481506, 125.0997009
28: -55.3510818, 48.7027550, -55.1145325, 48.6033134, -103.9543915, 103.8172760
29: -89.3067932, 49.5889664, -89.1124573, 49.2797546, -138.5865479, 138.7014160
30: -67.6956024, 60.4633713, -67.5522537, 60.2392807, -127.9348831, 128.0156250
31: -71.8636475, 48.1139297, -71.5968170, 47.9929199, -119.8565674, 119.7107468
32: -75.1507568, 48.1703606, -75.0070343, 47.8587952, -123.0095367, 123.1773834
33: -97.6053925, 69.0429688, -97.1052399, 68.9231720, -166.5285645, 166.1482086
34: -81.2449036, 54.3058510, -80.9523926, 54.2012787, -135.4461823, 135.2582397
35: -84.0147705, 57.1411209, -83.5387497, 57.0546494, -141.0693970, 140.6798706
36: -82.1383209, 56.2309952, -81.8634949, 56.1118698, -138.2501831, 138.0944824
37: -114.5621948, 58.0878601, -114.2252502, 57.9223328, -172.4845276, 172.3131104
38: -100.2545853, 68.5001984, -99.8996582, 68.3540039, -168.6085815, 168.3998566
39: -117.2774658, 67.2424774, -116.8689423, 67.1679916, -184.4454651, 184.1114197
40: -97.0952606, 53.1780624, -96.7294617, 53.1348114, -150.2300568, 149.9075317
41: -72.5276184, 46.7146645, -72.3625641, 46.5618057, -119.0894165, 119.0772247
42: -54.3819733, 43.2877846, -54.2802391, 42.7130051, -97.0949783, 97.5680237

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1719
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 116.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 103.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -99.8800507, 48.4665413, -99.8083878, 48.4708862, -148.3509216, 148.2749329
1: -56.4825974, 47.2755089, -56.4171753, 47.3414001, -103.8239975, 103.6926727
2: -48.4880447, 44.4165649, -48.4371681, 44.4113235, -92.8993683, 92.8537292
3: -53.5795288, 56.2886772, -53.5130424, 56.2880859, -109.8676071, 109.8017120
4: -57.3572845, 55.2825012, -57.3404083, 55.2933311, -112.6506042, 112.6229095
5: -52.0902023, 57.2825165, -52.0211639, 57.2855911, -109.3757935, 109.3036652
6: -70.0606461, 50.3325424, -70.0877686, 50.3846397, -120.4452820, 120.4203033
7: -64.7935791, 59.1525650, -64.7292252, 59.2506981, -124.0442810, 123.8817902
8: -63.6281357, 68.7611237, -63.5907135, 68.7753448, -132.4034729, 132.3518372
9: -59.1659088, 50.9094162, -59.1585693, 50.9453964, -110.1113052, 110.0679855
10: -83.5302734, 77.2529907, -83.5321350, 77.3227081, -160.8529663, 160.7851257
11: -78.6010437, 59.5979309, -78.6539917, 59.6684265, -138.2694702, 138.2519226
12: -77.3637314, 64.6146240, -77.4109650, 64.6404114, -142.0041504, 142.0255890
13: -79.9637146, 73.0558319, -79.9813843, 73.2260895, -153.1898041, 153.0372009
14: -121.7508240, 60.8251266, -121.7901154, 60.9940796, -182.7449036, 182.6152344
15: -65.8279343, 49.3760910, -65.8662109, 49.3853340, -115.2132568, 115.2423019
16: -87.7505264, 59.5752335, -87.7348022, 59.6479912, -147.3984985, 147.3100281
17: -122.8125763, 91.6054688, -122.8858795, 91.8020020, -214.6145782, 214.4913483
18: -70.3662109, 59.6748276, -70.4666901, 59.5876732, -129.9538727, 130.1415100
19: -55.3375854, 35.2297554, -55.3576927, 35.2275124, -90.5650940, 90.5874481
20: -50.4838028, 42.5601807, -50.4951706, 42.5696487, -93.0534515, 93.0553436
21: -71.3183441, 45.8597145, -71.3471985, 45.8523712, -117.1707001, 117.2069092
22: -77.8317871, 48.2006302, -77.9075394, 48.1449509, -125.9767303, 126.1081543
23: -56.5988693, 45.0576935, -56.6171379, 45.0698700, -101.6687393, 101.6748352
24: -70.5475388, 51.9025726, -70.6248016, 51.8998260, -122.4473648, 122.5273666
25: -58.5172081, 52.1451988, -58.5539398, 52.1767960, -110.6940002, 110.6991425
26: -77.7347717, 69.8144073, -77.7713013, 69.6847763, -147.4195557, 147.5857086
27: -74.4454346, 51.1497803, -74.5348587, 51.0934143, -125.5388489, 125.6846390
28: -55.3510818, 48.7027550, -55.3639984, 48.6994133, -104.0504837, 104.0667496
29: -89.3067932, 49.5889664, -89.3207703, 49.5324974, -138.8392792, 138.9097290
30: -67.6956024, 60.4633713, -67.7012863, 60.4526558, -128.1482391, 128.1646576
31: -71.8636475, 48.1139297, -71.8739777, 48.1067429, -119.9703903, 119.9878998
32: -75.1507568, 48.1703606, -75.1643066, 48.1677856, -123.3185425, 123.3346710
33: -97.6053925, 69.0429688, -97.6834106, 69.0526276, -166.6580200, 166.7263794
34: -81.2449036, 54.3058510, -81.2811584, 54.3182373, -135.5631409, 135.5870056
35: -84.0147705, 57.1411209, -84.0599823, 57.1543503, -141.1691132, 141.2010956
36: -82.1383209, 56.2309952, -82.1846466, 56.1873398, -138.3256531, 138.4156494
37: -114.5621948, 58.0878601, -114.7038345, 58.0123901, -172.5745850, 172.7916870
38: -100.2545853, 68.5001984, -100.2653656, 68.5293961, -168.7839813, 168.7655640
39: -117.2774658, 67.2424774, -117.3619385, 67.2419739, -184.5194397, 184.6044006
40: -97.0952606, 53.1780624, -97.2553635, 53.2038460, -150.2991028, 150.4334106
41: -72.5276184, 46.7146645, -72.5712585, 46.7134285, -119.2410431, 119.2859192
42: -54.3819733, 43.2877846, -54.3999481, 43.3079720, -97.6899414, 97.6877289

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1719
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1639
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

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 357.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
time: 121.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -99.5856476, 48.3846588, -99.8294067, 48.3961792, -147.9818268, 148.2140503
1: -56.3086967, 47.2162971, -56.4514656, 47.2142868, -103.5229721, 103.6677551
2: -48.2131691, 44.3498802, -48.3335114, 44.4094124, -92.6225739, 92.6833954
3: -53.2788467, 56.1954002, -53.5409393, 56.2515030, -109.5303497, 109.7363434
4: -57.0230331, 55.2006760, -57.2403908, 55.2862473, -112.3092804, 112.4410629
5: -51.8289413, 57.1844406, -52.0592346, 57.2594299, -109.0883484, 109.2436676
6: -69.9569626, 50.0662460, -70.0651779, 50.2779388, -120.2348785, 120.1314240
7: -64.6102066, 59.0837250, -64.6791992, 59.1454468, -123.7556458, 123.7629242
8: -63.3007050, 68.6743469, -63.4325447, 68.7319412, -132.0326538, 132.1068878
9: -58.9857864, 50.7713737, -59.2775726, 50.6882439, -109.6740265, 110.0489502
10: -83.3657379, 76.8933258, -83.4888382, 76.5961304, -159.9618530, 160.3821716
11: -78.4599838, 59.0493240, -78.5924301, 59.4089088, -137.8688965, 137.6417542
12: -77.2385101, 64.0243073, -77.3830414, 64.2266235, -141.4651337, 141.4073486
13: -79.8694305, 72.8222504, -79.9520035, 72.9284592, -152.7978821, 152.7742310
14: -121.5835571, 60.4456673, -121.7174530, 60.5765839, -182.1601410, 182.1631165
15: -65.4655075, 49.3298950, -65.8662415, 49.3181915, -114.7836990, 115.1961365
16: -87.5752106, 59.2739182, -87.6947479, 59.2746506, -146.8498535, 146.9686584
17: -122.6262894, 90.8680115, -122.8890533, 91.3792114, -214.0054932, 213.7570648
18: -70.2053146, 59.4807243, -70.2812195, 59.6040878, -129.8093872, 129.7619324
19: -55.2188644, 35.0055771, -55.3018875, 35.4046402, -90.6235046, 90.3074646
20: -50.3755684, 42.3596497, -50.4063492, 42.6429596, -93.0185242, 92.7659988
21: -71.1913452, 45.5504913, -71.2911072, 46.0346451, -117.2259903, 116.8415985
22: -77.6941452, 47.9435120, -77.6562576, 48.2573624, -125.9515076, 125.5997620
23: -56.5055695, 44.8930283, -56.5814590, 45.1227188, -101.6282883, 101.4744873
24: -70.3917236, 51.8430748, -70.3060760, 51.8946991, -122.2864227, 122.1491547
25: -58.4175758, 51.9816170, -58.4560089, 52.2213249, -110.6389008, 110.4376221
26: -77.5940399, 69.4550629, -77.6903229, 69.7355499, -147.3295746, 147.1453857
27: -74.2639160, 51.0392303, -74.0631561, 51.1243324, -125.3882446, 125.1023865
28: -55.2340622, 48.5295715, -55.2121277, 48.8141327, -104.0481873, 103.7416992
29: -89.1900940, 49.2333145, -89.2179642, 49.7014160, -138.8915100, 138.4512634
30: -67.6101685, 60.2193947, -67.6352692, 60.5227509, -128.1329041, 127.8546600
31: -71.7258835, 47.9440384, -71.7204361, 48.1994514, -119.9253387, 119.6644745
32: -75.0644836, 47.9039307, -75.0843430, 48.1447296, -123.2092133, 122.9882736
33: -97.2782440, 68.9443970, -97.4159698, 69.0236816, -166.3019257, 166.3603516
34: -80.9784546, 54.2165565, -81.2352448, 54.2932091, -135.2716675, 135.4517975
35: -83.7603302, 57.0805588, -83.7633972, 57.1127930, -140.8731232, 140.8439636
36: -82.0116425, 56.0584793, -81.9584808, 56.3214226, -138.3330688, 138.0169678
37: -114.3613892, 57.9312820, -114.3890305, 58.1016426, -172.4630280, 172.3203125
38: -100.0243835, 68.3783798, -100.1177826, 68.4801407, -168.5045166, 168.4961548
39: -117.0277710, 67.1780853, -117.0878067, 67.2288055, -184.2565765, 184.2658844
40: -96.8575363, 53.1103287, -96.9386292, 53.2059174, -150.0634460, 150.0489502
41: -72.4134827, 46.5552292, -72.4695740, 46.7409706, -119.1544495, 119.0247955
42: -54.2885475, 42.9714928, -54.3738098, 43.0053635, -97.2939148, 97.3453064

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 537
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
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1574
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1580
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1639
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5523038
time: 99.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5523038
time: 109.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -99.6422348, 48.4110603, -100.1333466, 48.5463638, -148.1885986, 148.5444031
1: -56.3389130, 47.2463036, -56.6127434, 47.3854141, -103.7243195, 103.8590469
2: -48.2806168, 44.3637619, -48.7173538, 44.4785614, -92.7591782, 93.0811157
3: -53.3354645, 56.2194977, -53.8453217, 56.3773117, -109.7127686, 110.0648193
4: -57.1018524, 55.2226562, -57.6913376, 55.3691673, -112.4710083, 112.9139862
5: -51.8756332, 57.2097778, -52.3131676, 57.3780365, -109.2536697, 109.5229492
6: -69.9912796, 50.1353760, -70.1813812, 50.6589928, -120.6502686, 120.3167572
7: -64.6530609, 59.1130905, -64.9181595, 59.3098602, -123.9629211, 124.0312424
8: -63.3857040, 68.6983566, -63.9202881, 68.8567657, -132.2424622, 132.6186523
9: -59.0058174, 50.8339119, -59.3838921, 51.0474396, -110.0532532, 110.2178040
10: -83.4045258, 77.0621796, -83.7057266, 77.5846329, -160.9891663, 160.7678986
11: -78.4965744, 59.1900139, -78.7890091, 60.2209435, -138.7175140, 137.9790192
12: -77.2656708, 64.1919861, -77.5308685, 65.2147369, -142.4804077, 141.7228394
13: -79.8949814, 72.9078064, -80.0742188, 73.4286194, -153.3235931, 152.9820251
14: -121.6271667, 60.5731506, -121.9530258, 61.3398972, -182.9670715, 182.5261841
15: -65.5419922, 49.3656845, -66.2750168, 49.4291458, -114.9711151, 115.6407013
16: -87.6220703, 59.3837051, -87.9146881, 59.9040298, -147.5260925, 147.2983856
17: -122.6604767, 91.0631104, -123.0756683, 92.5413361, -215.2018127, 214.1387634
18: -70.2627640, 59.5181656, -70.6087952, 59.8065758, -130.0693359, 130.1269531
19: -55.2506409, 35.0257835, -55.4774590, 35.5039978, -90.7546387, 90.5032349
20: -50.4093399, 42.3887787, -50.5947685, 42.8022842, -93.2116241, 92.9835510
21: -71.2238770, 45.5859070, -71.4697266, 46.2238312, -117.4477005, 117.0556335
22: -77.7593842, 47.9861069, -78.0151138, 48.4412994, -126.2006836, 126.0012207
23: -56.5302887, 44.9199371, -56.7061768, 45.2606430, -101.7909317, 101.6261063
24: -70.4662933, 51.8561401, -70.7350616, 51.9621811, -122.4284744, 122.5912018
25: -58.4523201, 52.0072784, -58.6437492, 52.3665085, -110.8188324, 110.6510315
26: -77.6328659, 69.5193329, -77.9077301, 70.0893097, -147.7221680, 147.4270630
27: -74.3629074, 51.0571251, -74.6479797, 51.2155228, -125.5784302, 125.7051086
28: -55.2771492, 48.5479431, -55.4624329, 48.9103088, -104.1874542, 104.0103760
29: -89.2277832, 49.2786942, -89.4270172, 49.9543915, -139.1821747, 138.7057190
30: -67.6430664, 60.2611618, -67.7831879, 60.7371101, -128.3801727, 128.0443420
31: -71.7763367, 47.9661102, -71.9996033, 48.3134117, -120.0897446, 119.9657135
32: -75.0932159, 47.9593964, -75.2412186, 48.4541702, -123.5473862, 123.2006073
33: -97.3785782, 68.9677277, -97.9945679, 69.1525116, -166.5310974, 166.9622955
34: -81.0367661, 54.2382431, -81.5644836, 54.4097595, -135.4465179, 135.8027344
35: -83.8510590, 57.0984535, -84.2850647, 57.2120972, -141.0631561, 141.3835144
36: -82.0687637, 56.0769424, -82.2799835, 56.3965797, -138.4653473, 138.3569336
37: -114.4458008, 57.9527664, -114.8695679, 58.1924171, -172.6382141, 172.8223267
38: -100.0925293, 68.4101562, -100.4847488, 68.6559143, -168.7484436, 168.8948975
39: -117.1203003, 67.1925964, -117.5827942, 67.3027954, -184.4230957, 184.7753906
40: -96.9475098, 53.1239548, -97.4655685, 53.2752228, -150.2227173, 150.5895233
41: -72.4517670, 46.5846481, -72.6785965, 46.8931961, -119.3449631, 119.2632446
42: -54.3113708, 43.0742340, -54.4933014, 43.6011276, -97.9124985, 97.5675354

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 537
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
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1574
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
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1253
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1639
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.6120610
time: 135.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.6120612
time: 119.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -99.9095230, 48.4571838, -99.8294067, 48.3961792, -148.3056946, 148.2865906
1: -56.5044861, 47.2591095, -56.4514656, 47.2142868, -103.7187653, 103.7105713
2: -48.4947510, 44.4165916, -48.3335114, 44.4094124, -92.9041595, 92.7501068
3: -53.6121101, 56.2839737, -53.5409393, 56.2515030, -109.8636093, 109.8249130
4: -57.3753433, 55.2755661, -57.2403908, 55.2862473, -112.6615906, 112.5159531
5: -52.1219101, 57.2765427, -52.0592346, 57.2594299, -109.3813400, 109.3357773
6: -70.0495911, 50.3404274, -70.0651779, 50.2779388, -120.3275299, 120.4056015
7: -64.8020782, 59.1417580, -64.6791992, 59.1454468, -123.9475250, 123.8209534
8: -63.6316643, 68.7550125, -63.4325447, 68.7319412, -132.3635864, 132.1875610
9: -59.2032700, 50.8726578, -59.2775726, 50.6882439, -109.8915100, 110.1502304
10: -83.5344315, 77.1559143, -83.4888382, 76.5961304, -160.1305542, 160.6447449
11: -78.5925903, 59.6040268, -78.5924301, 59.4089088, -138.0014954, 138.1964417
12: -77.3586273, 64.6004791, -77.3830414, 64.2266235, -141.5852509, 141.9835205
13: -79.9628601, 73.0218811, -79.9520035, 72.9284592, -152.8913269, 152.9738770
14: -121.7469635, 60.7897797, -121.7174530, 60.5765839, -182.3235474, 182.5072021
15: -65.8530426, 49.3654480, -65.8662415, 49.3181915, -115.1712341, 115.2316895
16: -87.7497406, 59.5238342, -87.6947479, 59.2746506, -147.0243835, 147.2185822
17: -122.8168945, 91.6063614, -122.8890533, 91.3792114, -214.1960907, 214.4954071
18: -70.3439713, 59.6975365, -70.2812195, 59.6040878, -129.9480591, 129.9787598
19: -55.3333435, 35.2827911, -55.3018875, 35.4046402, -90.7379761, 90.5846786
20: -50.4743118, 42.5918694, -50.4063492, 42.6429596, -93.1172638, 92.9982147
21: -71.3125610, 45.9220505, -71.2911072, 46.0346451, -117.3472061, 117.2131577
22: -77.8009796, 48.2364960, -77.6562576, 48.2573624, -126.0583420, 125.8927460
23: -56.5947342, 45.0833664, -56.5814590, 45.1227188, -101.7174530, 101.6648254
24: -70.5016098, 51.9061050, -70.3060760, 51.8946991, -122.3963089, 122.2121811
25: -58.5065002, 52.1695404, -58.4560089, 52.2213249, -110.7278290, 110.6255493
26: -77.7296143, 69.8571777, -77.6903229, 69.7355499, -147.4651337, 147.5474854
27: -74.3780899, 51.1649780, -74.0631561, 51.1243324, -125.5024261, 125.2281342
28: -55.3305168, 48.7402534, -55.2121277, 48.8141327, -104.1446381, 103.9523773
29: -89.2957458, 49.6542816, -89.2179642, 49.7014160, -138.9971619, 138.8722382
30: -67.6842041, 60.4989319, -67.6352692, 60.5227509, -128.2069397, 128.1342010
31: -71.8466339, 48.1464920, -71.7204361, 48.1994514, -120.0460815, 119.8669205
32: -75.1407166, 48.1917038, -75.0843430, 48.1447296, -123.2854462, 123.2760315
33: -97.5881500, 69.0439224, -97.4159698, 69.0236816, -166.6118164, 166.4598846
34: -81.2611237, 54.3073807, -81.2352448, 54.2932091, -135.5543213, 135.5426331
35: -83.9850693, 57.1377716, -83.7633972, 57.1127930, -141.0978699, 140.9011688
36: -82.1065063, 56.2660370, -81.9584808, 56.3214226, -138.4279175, 138.2245178
37: -114.5224228, 58.1126328, -114.3890305, 58.1016426, -172.6240692, 172.5016479
38: -100.2436066, 68.5024948, -100.1177826, 68.4801407, -168.7237396, 168.6202698
39: -117.2453232, 67.2385330, -117.0878067, 67.2288055, -184.4741211, 184.3263397
40: -97.0627975, 53.1824341, -96.9386292, 53.2059174, -150.2687073, 150.1210632
41: -72.5190659, 46.7323837, -72.4695740, 46.7409706, -119.2600403, 119.2019501
42: -54.3806877, 43.2658310, -54.3738098, 43.0053635, -97.3860474, 97.6396408

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1720
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
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1090
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1639
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.4702898
time: 121.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702893
time: 124.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -99.9662552, 48.4835434, -100.1333466, 48.5463638, -148.5126038, 148.6168823
1: -56.5347748, 47.2891769, -56.6127434, 47.3854141, -103.9201889, 103.9019089
2: -48.5622482, 44.4304543, -48.7173538, 44.4785614, -93.0408096, 93.1478043
3: -53.6687622, 56.3080788, -53.8453217, 56.3773117, -110.0460663, 110.1533966
4: -57.4542198, 55.2975311, -57.6913376, 55.3691673, -112.8233795, 112.9888611
5: -52.1687775, 57.3017807, -52.3131676, 57.3780365, -109.5468140, 109.6149445
6: -70.0837708, 50.4099388, -70.1813812, 50.6589928, -120.7427673, 120.5913239
7: -64.8451538, 59.1712074, -64.9181595, 59.3098602, -124.1550140, 124.0893555
8: -63.7167282, 68.7790527, -63.9202881, 68.8567657, -132.5734863, 132.6993408
9: -59.2234421, 50.9355545, -59.3838921, 51.0474396, -110.2708740, 110.3194427
10: -83.5734787, 77.3249969, -83.7057266, 77.5846329, -161.1581116, 161.0307007
11: -78.6290283, 59.7447052, -78.7890091, 60.2209435, -138.8499756, 138.5337219
12: -77.3857269, 64.7681885, -77.5308685, 65.2147369, -142.6004639, 142.2990417
13: -79.9884872, 73.1077118, -80.0742188, 73.4286194, -153.4170990, 153.1819153
14: -121.7904892, 60.9173088, -121.9530258, 61.3398972, -183.1303864, 182.8703308
15: -65.9314423, 49.4010239, -66.2750168, 49.4291458, -115.3605804, 115.6760406
16: -87.7965546, 59.6336212, -87.9146881, 59.9040298, -147.7005768, 147.5483093
17: -122.8510056, 91.8015747, -123.0756683, 92.5413361, -215.3923340, 214.8772430
18: -70.4017334, 59.7351608, -70.6087952, 59.8065758, -130.2083130, 130.3439484
19: -55.3653107, 35.3030167, -55.4774590, 35.5039978, -90.8693085, 90.7804718
20: -50.5081139, 42.6210480, -50.5947685, 42.8022842, -93.3103943, 93.2158203
21: -71.3450317, 45.9575195, -71.4697266, 46.2238312, -117.5688553, 117.4272308
22: -77.8663635, 48.2790604, -78.0151138, 48.4412994, -126.3076630, 126.2941742
23: -56.6193657, 45.1103554, -56.7061768, 45.2606430, -101.8800049, 101.8165283
24: -70.5762634, 51.9191780, -70.7350616, 51.9621811, -122.5384369, 122.6542358
25: -58.5413437, 52.1953201, -58.6437492, 52.3665085, -110.9078522, 110.8390656
26: -77.7685089, 69.9219513, -77.9077301, 70.0893097, -147.8578033, 147.8296661
27: -74.4770432, 51.1829147, -74.6479797, 51.2155228, -125.6925659, 125.8308945
28: -55.3737106, 48.7586746, -55.4624329, 48.9103088, -104.2840195, 104.2210999
29: -89.3337021, 49.6997147, -89.4270172, 49.9543915, -139.2880859, 139.1267242
30: -67.7169495, 60.5408707, -67.7831879, 60.7371101, -128.4540558, 128.3240509
31: -71.8977127, 48.1686745, -71.9996033, 48.3134117, -120.2111206, 120.1682739
32: -75.1693954, 48.2472610, -75.2412186, 48.4541702, -123.6235657, 123.4884796
33: -97.6885834, 69.0671692, -97.9945679, 69.1525116, -166.8410950, 167.0617371
34: -81.3195190, 54.3290405, -81.5644836, 54.4097595, -135.7292786, 135.8935242
35: -84.0758820, 57.1555748, -84.2850647, 57.2120972, -141.2879791, 141.4406281
36: -82.1636810, 56.2845230, -82.2799835, 56.3965797, -138.5602570, 138.5645142
37: -114.6070175, 58.1342697, -114.8695679, 58.1924171, -172.7994232, 173.0038452
38: -100.3118896, 68.5343246, -100.4847488, 68.6559143, -168.9678040, 169.0190582
39: -117.3386536, 67.2530746, -117.5827942, 67.3027954, -184.6414490, 184.8358612
40: -97.1529541, 53.1961365, -97.4655685, 53.2752228, -150.4281616, 150.6617126
41: -72.5574036, 46.7620697, -72.6785965, 46.8931961, -119.4505997, 119.4406662
42: -54.4034958, 43.3686600, -54.4933014, 43.6011276, -98.0046158, 97.8619614

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1639
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5307437
time: 148.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5307432
time: 125.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -99.8253021, 48.4735718, -99.5785980, 48.3169632, -148.1422424, 148.0521698
1: -56.4199944, 47.3457184, -56.3216705, 47.1047134, -103.5247040, 103.6673813
2: -48.4344788, 44.4187241, -48.1051064, 44.3471794, -92.7816620, 92.5238342
3: -53.5325470, 56.2930908, -53.2762527, 56.1626205, -109.6951675, 109.5693359
4: -57.3650055, 55.2925568, -56.9068680, 55.1990891, -112.5640717, 112.1994247
5: -52.0352936, 57.2940063, -51.8372765, 57.1639175, -109.1991959, 109.1312866
6: -70.0862198, 50.4245300, -69.9438934, 49.9536133, -120.0398331, 120.3684235
7: -64.7269745, 59.2724915, -64.5550537, 58.9895859, -123.7165604, 123.8275452
8: -63.5927124, 68.7822571, -63.1415291, 68.6350784, -132.2277832, 131.9237823
9: -59.1997223, 50.9240112, -59.0601578, 50.5521278, -109.7518387, 109.9841614
10: -83.5426025, 77.2640152, -83.3141098, 76.2665405, -159.8091278, 160.5781250
11: -78.6631012, 59.7204285, -78.4008713, 58.7867661, -137.4498596, 138.1213074
12: -77.4210358, 64.6531448, -77.2147827, 63.6275826, -141.0486145, 141.8679199
13: -79.9898071, 73.2634659, -79.8407211, 72.5586166, -152.5484161, 153.1041870
14: -121.7966156, 61.0300026, -121.5140152, 60.0625000, -181.8591156, 182.5440216
15: -65.9249115, 49.3863983, -65.4253845, 49.2643814, -115.1892929, 114.8117828
16: -87.7463837, 59.6423531, -87.5320053, 58.9461670, -146.6925354, 147.1743469
17: -122.9033508, 91.8981628, -122.6249084, 90.4447632, -213.3481140, 214.5230713
18: -70.4807510, 59.5880661, -70.0420380, 59.4734612, -129.9541931, 129.6300964
19: -55.3639526, 35.2934189, -55.1622162, 35.1307755, -90.4947205, 90.4556274
20: -50.4932022, 42.6111298, -50.2969284, 42.4016800, -92.8948822, 92.9080429
21: -71.3540878, 45.9227257, -71.1392517, 45.6715851, -117.0256500, 117.0619736
22: -77.9026794, 48.1658859, -77.4739380, 48.0171623, -125.9198380, 125.6398239
23: -56.6216507, 45.1054688, -56.4725838, 44.9204102, -101.5420609, 101.5780487
24: -70.6145325, 51.9029999, -70.1193390, 51.8350334, -122.4495544, 122.0223389
25: -58.5607376, 52.2182770, -58.3300552, 52.0011406, -110.5618744, 110.5483322
26: -77.7794800, 69.6924973, -77.5183411, 69.4642487, -147.2437286, 147.2108459
27: -74.5026474, 51.0990639, -73.8632660, 51.0587311, -125.5613785, 124.9623260
28: -55.3497543, 48.7454910, -55.1011124, 48.6067581, -103.9565125, 103.8465958
29: -89.3202515, 49.5887604, -89.0983429, 49.3371582, -138.6574097, 138.6870880
30: -67.7043762, 60.4828529, -67.5469971, 60.2502403, -127.9546204, 128.0298309
31: -71.8728943, 48.1425323, -71.5860291, 48.0011139, -119.8740005, 119.7285614
32: -75.1632614, 48.1977921, -74.9932709, 47.8617630, -123.0249939, 123.1910629
33: -97.7030640, 69.0582809, -97.0272827, 68.9131622, -166.6162262, 166.0855713
34: -81.3184204, 54.3256493, -80.9168854, 54.1888275, -135.5072479, 135.2425232
35: -84.0533447, 57.1562042, -83.4939270, 57.0411072, -141.0944214, 140.6501160
36: -82.1695099, 56.2124176, -81.8175354, 56.1556435, -138.3251495, 138.0299530
37: -114.7227173, 58.0153122, -114.0855560, 57.9970398, -172.7197571, 172.1008606
38: -100.2564087, 68.5489502, -99.8896790, 68.3253479, -168.5817413, 168.4386292
39: -117.3678131, 67.2416687, -116.7845154, 67.1682663, -184.5360718, 184.0261688
40: -97.2868423, 53.2125130, -96.5696487, 53.1083946, -150.3952332, 149.7821655
41: -72.5770798, 46.7425003, -72.3190536, 46.5637627, -119.1408386, 119.0615540
42: -54.4055824, 43.3059921, -54.2617531, 42.6929092, -97.0984955, 97.5677490

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 533
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
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1719
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
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1702
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
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4239088, upper bound: 69.4862740
time: 116.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -69.4239088, upper bound: 69.4862740
time: 849.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 968.30 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.5192941
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.5192941
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4362265, upper bound: 69.4573688
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5523038
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5523038
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.6120610
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.6120612
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.4702898
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4702893
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5307437
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4188942, upper bound: 69.5307432
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4239088, upper bound: 69.4862740
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 968.30
Output dim: 4, lower bound: -69.4239088, upper bound: 69.4862740
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.4892908, upper bound: 69.5859835
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.6197169, upper bound: 69.5347599
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.6218561, upper bound: 69.5968062
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.4833982, upper bound: 69.4927358
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.5588825, upper bound: 69.5866541
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.5938845, upper bound: 69.5357092
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 968.30
Output dim: 4, lower bound: -69.5968062, upper bound: 69.5978847

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 119.81 + 7741.25 = 7861.06 seconds
