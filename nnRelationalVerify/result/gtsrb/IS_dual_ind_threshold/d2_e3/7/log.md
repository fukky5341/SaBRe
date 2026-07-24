## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 9.2022701184


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.9057083, 21.9057083)
1: (-0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1723938, 14.1723938)
2: (-1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3840714, 15.3840714)
3: (-7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6941528, 16.6941528)
4: (-4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3889160, 18.3889198)
5: (-6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.5073013, 18.5073051)
6: (-32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0350189, 21.0350227)
7: (-5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3901405, 16.3901443)
8: (-16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6367073, 20.6367035)
9: (-5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5841675, 21.5841675)
10: (-21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4209595, 27.4209671)
11: (-16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0891457, 18.0891418)
12: (-35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5512009, 26.5512009)
13: (-21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9042969, 34.9042969)
14: (-52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8795242, 27.8795242)
15: (-13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7997131, 18.7997131)
16: (-13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8603897, 22.8603897)
17: (-46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7479553, 28.7479477)
18: (-17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4578171, 25.4578171)
19: (-16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0739975, 17.0739975)
20: (-19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3917885, 17.3917885)
21: (-19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1010590, 19.1010590)
22: (-19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8323593, 18.8323593)
23: (-15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5344772, 19.5344734)
24: (-14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6371994, 20.6371994)
25: (-19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3379135, 20.3379135)
26: (-33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4708405, 27.4708405)
27: (-16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873)
28: (-18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1714325, 22.1714325)
29: (-17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5099182, 17.5099182)
30: (-24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1623993, 24.1623993)
31: (-17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7010956, 18.7010956)
32: (-24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1186523, 23.1186600)
33: (-40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.2023773, 32.2023773)
34: (-40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6639328, 24.6639404)
35: (-30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5269470, 24.5269470)
36: (-30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9577026, 28.9577026)
37: (-47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8535767, 33.8535690)
38: (-38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6222229, 37.6222229)
39: (-42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3223114, 39.3223114)
40: (-39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7966919, 24.7966919)
41: (-22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5928650, 24.5928726)
42: (-24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8311577, 15.8311615)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.40 + 41.12 = 43.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -9.2114816, upper bound: 9.2114816

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2050112, upper bound: 9.1839168
time: 9.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2050112, upper bound: 9.2050110
time: 28.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 37.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 37.82
Output dim: 2, lower bound: -9.2050112, upper bound: 9.1839168
IS_A2, status: Status.UNKNOWN, split count: 1, time: 37.82
Output dim: 2, lower bound: -9.2050112, upper bound: 9.2050110

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.4473972, 17.7597809, -7.4498196, 17.7614174, -21.8998413, 21.8354645
1: -0.8150017, 19.0368729, -0.8160541, 19.0383606, -14.1693840, 14.1260719
2: -1.2606616, 18.9822502, -1.2620556, 18.9985199, -15.3708305, 15.3507538
3: -7.5761843, 15.5515499, -7.5790472, 15.5542030, -16.6865501, 16.7283020
4: -4.9506960, 17.5605259, -4.9528294, 17.5699635, -18.4106712, 18.3799210
5: -6.5316987, 18.0290184, -6.5340924, 18.0340328, -18.5000153, 18.5078239
6: -32.9257736, -5.5566788, -32.9544296, -5.5542936, -20.9844666, 20.9043427
7: -5.7592983, 18.0725250, -5.7608919, 18.0745811, -16.3853645, 16.2749176
8: -16.5298157, 8.8372431, -16.5310669, 8.8478012, -20.6285362, 20.6154366
9: -5.6627645, 19.6471882, -5.6680603, 19.6493816, -21.5749893, 21.6337280
10: -21.8808575, 13.0031223, -21.8837643, 13.0159121, -27.3699265, 27.3771362
11: -16.6924305, 5.9544830, -16.6951427, 5.9571943, -18.0793724, 18.1050835
12: -35.2401733, -3.6053243, -35.2492867, -3.6015058, -26.5145798, 26.5401688
13: -21.3068466, 16.2839355, -21.3102722, 16.2881813, -34.8935394, 34.8894501
14: -52.7755623, -14.7970343, -52.7782631, -14.7635813, -27.5205536, 27.8202362
15: -13.0231123, 11.8782902, -13.0255728, 11.8796844, -18.7281036, 18.7949753
16: -13.0858574, 16.0986443, -13.0891018, 16.1015396, -22.8514404, 22.9179764
17: -46.4462395, -12.0728149, -46.4495468, -12.0364542, -28.4339752, 28.6825790
18: -17.1448097, 11.4582281, -17.1480923, 11.4626322, -25.5671463, 25.4442749
19: -16.7551155, 6.1193838, -16.7580299, 6.1201081, -17.0703430, 17.0664024
20: -19.5903587, 2.8865137, -19.5920620, 2.8878884, -17.3720016, 17.3849258
21: -19.3997288, 5.4644175, -19.4024925, 5.4669833, -19.0918198, 19.1035385
22: -19.9429951, 6.1004930, -19.9478607, 6.1014838, -18.7523575, 18.8260803
23: -15.8723373, 9.2902126, -15.8742113, 9.2921877, -19.5290756, 19.5286942
24: -14.6680813, 9.6909523, -14.6701584, 9.6993895, -20.6456757, 20.6223068
25: -19.4747162, 5.8720074, -19.4764347, 5.8805985, -20.2914124, 20.3183975
26: -33.1906204, 4.2765155, -33.1944847, 4.2788363, -27.4592285, 27.4633789
27: -16.3879852, 11.1814957, -16.4043427, 11.1835918, -27.5715771, 27.5858383
28: -18.5310707, 9.6033669, -18.5368366, 9.6049633, -22.1265640, 22.1638641
29: -17.3881397, 3.6740565, -17.3945332, 3.6747675, -17.4601746, 17.5041084
30: -24.8439827, 1.7474160, -24.8450928, 1.7508812, -24.1068802, 24.1438522
31: -17.8071632, 6.6202650, -17.8089752, 6.6273680, -18.6961212, 18.6918831
32: -24.0476418, 2.6605663, -24.0767708, 2.6632628, -23.0671692, 23.0879974
33: -40.9096756, -2.0228767, -40.9124603, -2.0203323, -32.1923981, 32.1882324
34: -40.1099434, -9.4783373, -40.1296616, -9.4776173, -24.6380539, 24.6484299
35: -30.2069054, 0.4595861, -30.2147942, 0.4600353, -24.5385590, 24.5197830
36: -30.0582085, 4.7502017, -30.0885010, 4.7508249, -28.9065247, 28.9352493
37: -47.5329895, -6.8155203, -47.5509109, -6.8134618, -33.8199768, 33.8104706
38: -38.0971222, 4.7409496, -38.1246033, 4.7420292, -37.5971985, 37.5996857
39: -42.1456566, -1.3242731, -42.1503220, -1.3222446, -39.3036804, 39.3057404
40: -39.7142715, -10.9534922, -39.7319641, -10.9513569, -24.7642288, 24.7567978
41: -22.8181286, 6.2598662, -22.8540192, 6.2620754, -24.5299225, 24.4615326
42: -24.6102314, -2.2427986, -24.6329422, -2.2406142, -15.7937775, 15.8115654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=153, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1735905, upper bound: 9.1826754
time: 31.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2037921, upper bound: 9.1826754
time: 28.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.5317783, 17.7756500, -7.4499760, 17.7617989, -21.9803696, 21.9377441
1: -0.8639290, 19.0470829, -0.8163624, 19.0385017, -14.2181625, 14.1932411
2: -1.3773117, 19.0112877, -1.2620709, 19.0073414, -15.4945831, 15.3787537
3: -7.6554785, 15.5609446, -7.5798140, 15.5545263, -16.7617340, 16.7025108
4: -5.0710716, 17.5766144, -4.9533434, 17.5743275, -18.4877014, 18.4085007
5: -6.6090217, 18.0348511, -6.5342274, 18.0328522, -18.5616760, 18.5116539
6: -32.9817505, -5.4037547, -32.9701385, -5.5540581, -21.0352173, 21.1842270
7: -5.8232384, 18.0798378, -5.7610774, 18.0751686, -16.4417763, 16.4099960
8: -16.6380329, 8.8627920, -16.5313454, 8.8530855, -20.7389755, 20.6578674
9: -5.6888561, 19.6938095, -5.6638546, 19.6494617, -21.6206360, 21.6037292
10: -21.9669418, 13.0402193, -21.8850422, 13.0182762, -27.5412979, 27.4050598
11: -16.7354317, 6.0184770, -16.6954098, 5.9582253, -18.1378403, 18.1439781
12: -35.2621956, -3.5114779, -35.2530670, -3.6013665, -26.5753403, 26.6051178
13: -21.3670349, 16.3070431, -21.3111515, 16.2889881, -34.9611359, 34.9215698
14: -52.9184761, -14.7366028, -52.7781715, -14.7451897, -28.0256042, 27.8716049
15: -13.1118078, 11.8843975, -13.0264807, 11.8798895, -18.8913422, 18.8013763
16: -13.1230850, 16.1524029, -13.0874891, 16.1018448, -22.8944397, 22.8998871
17: -46.6352997, -12.0114374, -46.4497795, -12.0163212, -28.9640579, 28.7259979
18: -17.1828346, 11.4893856, -17.1476746, 11.4621277, -25.4728699, 25.4932175
19: -16.7849255, 6.1437435, -16.7587662, 6.1202106, -17.0823364, 17.1177597
20: -19.5967712, 2.9163380, -19.5919132, 2.8883870, -17.4011269, 17.4125938
21: -19.4288769, 5.5114284, -19.4030113, 5.4680882, -19.1188965, 19.1648216
22: -19.9900436, 6.1159477, -19.9497719, 6.1017199, -18.8722916, 18.8530235
23: -15.8915205, 9.3112946, -15.8743725, 9.2927408, -19.5526123, 19.5541763
24: -14.7062654, 9.7172747, -14.6701012, 9.6987286, -20.6702118, 20.6512146
25: -19.5150242, 5.9121976, -19.4761658, 5.8813095, -20.3817902, 20.3587036
26: -33.2198410, 4.2977772, -33.1953125, 4.2793798, -27.4945831, 27.4910889
27: -16.4315338, 11.2685108, -16.4097614, 11.1841278, -27.6156616, 27.6782722
28: -18.5460014, 9.6522284, -18.5370026, 9.6053286, -22.1809006, 22.2146301
29: -17.4296608, 3.7219195, -17.3957310, 3.6746478, -17.5493546, 17.5519409
30: -24.8552437, 1.8021288, -24.8449936, 1.7518816, -24.1911545, 24.1806564
31: -17.8527451, 6.6517811, -17.8084717, 6.6283026, -18.7237701, 18.7452393
32: -24.1079292, 2.7780108, -24.0925713, 2.6630888, -23.1237488, 23.2377472
33: -40.9188614, -1.9561539, -40.9123573, -2.0193253, -32.2169800, 32.2650146
34: -40.1491699, -9.3875237, -40.1389961, -9.4779234, -24.6650925, 24.7419891
35: -30.2269821, 0.5131793, -30.2170181, 0.4602780, -24.5333862, 24.5842819
36: -30.1213131, 4.8710642, -30.1051769, 4.7507877, -28.9629059, 29.0767822
37: -47.5682564, -6.6925521, -47.5579834, -6.8131156, -33.8500061, 33.9657288
38: -38.1557999, 4.8613935, -38.1377525, 4.7420115, -37.6316986, 37.7442474
39: -42.1651993, -1.2600470, -42.1510773, -1.3220010, -39.3186188, 39.4001617
40: -39.7419701, -10.8207636, -39.7381363, -10.9513645, -24.7901917, 24.9178238
41: -22.8913383, 6.4038830, -22.8739662, 6.2615547, -24.5933304, 24.7365189
42: -24.6566277, -2.1210136, -24.6468582, -2.2409759, -15.8361969, 15.9546204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1735905, upper bound: 9.2037921
time: 29.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2037921, upper bound: 9.2037921
time: 25.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 57.00 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 57.00
Output dim: 2, lower bound: -9.1735905, upper bound: 9.1826754
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 57.00
Output dim: 2, lower bound: -9.2037921, upper bound: 9.1826754
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 57.00
Output dim: 2, lower bound: -9.1735905, upper bound: 9.2037921
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 57.00
Output dim: 2, lower bound: -9.2037921, upper bound: 9.2037921

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.4452801, 17.7581272, -7.5368156, 17.7625809, -21.8959160, 21.9231720
1: -0.8129730, 19.0351334, -0.8687634, 19.0401936, -14.1671600, 14.1789017
2: -1.2598932, 18.9793320, -1.3378146, 18.9994087, -15.3670845, 15.4231567
3: -7.5755811, 15.5498962, -7.6328125, 15.5575771, -16.6874466, 16.7791519
4: -4.9495883, 17.5575485, -5.0483932, 17.5686054, -18.4070587, 18.4729233
5: -6.5309649, 18.0265427, -6.5988626, 18.0366325, -18.4984703, 18.5704727
6: -32.9227028, -5.5578566, -32.9541473, -5.5149536, -21.0268555, 20.9037018
7: -5.7575588, 18.0706177, -5.8153844, 18.0759926, -16.3829727, 16.3277512
8: -16.5291824, 8.8328648, -16.6317177, 8.8508806, -20.6281395, 20.7116699
9: -5.6617284, 19.6454220, -5.7277780, 19.6484241, -21.5710068, 21.6942825
10: -21.8791161, 13.0002537, -21.9190483, 13.0214710, -27.3717194, 27.4107666
11: -16.6857204, 5.9537878, -16.6971436, 6.0395222, -18.1561050, 18.0981827
12: -35.2376976, -3.6064262, -35.2480850, -3.5245986, -26.5840378, 26.5477982
13: -21.3052483, 16.2805786, -21.3365364, 16.2981682, -34.9045258, 34.9122772
14: -52.7749023, -14.7980213, -52.8010559, -14.7418108, -27.5419846, 27.8435211
15: -13.0210247, 11.8752975, -13.1083927, 11.8792143, -18.7224808, 18.8783340
16: -13.0827093, 16.0973091, -13.1127377, 16.1070690, -22.8746796, 22.9172058
17: -46.4377251, -12.0741291, -46.4567108, -11.9646759, -28.5056152, 28.6894226
18: -17.1396122, 11.4577980, -17.1539555, 11.5458088, -25.6442642, 25.4449615
19: -16.7516422, 6.1189594, -16.7645073, 6.1479926, -17.0949554, 17.0697517
20: -19.5865135, 2.8853316, -19.5929546, 2.9253540, -17.3918686, 17.3869362
21: -19.3949718, 5.4636478, -19.4040680, 5.5186353, -19.1428757, 19.1013336
22: -19.9393234, 6.1000252, -19.9499474, 6.1205215, -18.7690964, 18.8269691
23: -15.8683462, 9.2896013, -15.8767939, 9.3499813, -19.5817642, 19.5275879
24: -14.6650753, 9.6904364, -14.6822987, 9.7655735, -20.7086334, 20.6291733
25: -19.4708405, 5.8710251, -19.4851418, 5.9585686, -20.3641205, 20.3230553
26: -33.1855659, 4.2759905, -33.1996841, 4.3639030, -27.5389404, 27.4615936
27: -16.3855343, 11.1810503, -16.4130821, 11.2032537, -27.5887871, 27.5941315
28: -18.5274010, 9.6027098, -18.5416069, 9.6713181, -22.1888428, 22.1637650
29: -17.3839493, 3.6737697, -17.3956146, 3.6910005, -17.4781036, 17.4985123
30: -24.8392353, 1.7462955, -24.8505211, 1.8592944, -24.2127380, 24.1421280
31: -17.8040829, 6.6192780, -17.8147659, 6.6490488, -18.7171249, 18.6968460
32: -24.0468636, 2.6590738, -24.1127129, 2.6915412, -23.0809097, 23.1393433
33: -40.9052963, -2.0250263, -40.9201889, -2.0034461, -32.2021179, 32.1991730
34: -40.1058731, -9.4806490, -40.1349831, -9.4131317, -24.6982727, 24.6452332
35: -30.2036209, 0.4578829, -30.2236977, 0.5072134, -24.5822754, 24.5242615
36: -30.0551929, 4.7491107, -30.0898247, 4.7874269, -28.9420929, 28.9346924
37: -47.5281372, -6.8163085, -47.5592651, -6.7585793, -33.8705902, 33.8157806
38: -38.0933685, 4.7385702, -38.1282120, 4.8065958, -37.6599426, 37.5988312
39: -42.1432114, -1.3274574, -42.1616440, -1.3176599, -39.3055115, 39.3131409
40: -39.7114182, -10.9567232, -39.7398987, -10.9442167, -24.7675934, 24.7596817
41: -22.8161755, 6.2595768, -22.8630142, 6.2806778, -24.5523987, 24.4730148
42: -24.6061268, -2.2438040, -24.6286774, -2.2165816, -15.8097916, 15.8211937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=153, inp2_unstable=155, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1836187, upper bound: 9.1810043
time: 29.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2029990, upper bound: 9.1818835
time: 38.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.5269299, 17.7688122, -7.4371986, 17.7442055, -21.9581337, 21.9185562
1: -0.8595281, 19.0397453, -0.8046963, 19.0191498, -14.1957207, 14.1751823
2: -1.3757432, 18.9990234, -1.2579129, 18.9748993, -15.4616699, 15.3627548
3: -7.6539645, 15.5558395, -7.5757518, 15.5410938, -16.7452774, 16.6920319
4: -5.0686383, 17.5647507, -4.9469953, 17.5429668, -18.4555244, 18.3912392
5: -6.6072540, 18.0257721, -6.5295563, 18.0087452, -18.5360184, 18.4973068
6: -32.9776764, -5.4059610, -32.9596634, -5.5597916, -21.0225601, 21.1665230
7: -5.8191857, 18.0717030, -5.7504635, 18.0537376, -16.4152718, 16.3911667
8: -16.6367722, 8.8460045, -16.5281219, 8.8087502, -20.6922035, 20.6374893
9: -5.6861110, 19.6866093, -5.6567903, 19.6303864, -21.5993652, 21.5904846
10: -21.9626961, 13.0327377, -21.8738232, 12.9988861, -27.5149994, 27.3854675
11: -16.7144241, 6.0170422, -16.6399155, 5.9543557, -18.1117630, 18.0827827
12: -35.2548180, -3.5139155, -35.2335434, -3.6078329, -26.5611725, 26.5857086
13: -21.3639793, 16.2992744, -21.3032074, 16.2685776, -34.9392853, 34.9057465
14: -52.9165039, -14.7393646, -52.7730675, -14.7526588, -28.0132751, 27.8612900
15: -13.1075172, 11.8728714, -13.0152416, 11.8494635, -18.8552933, 18.7784119
16: -13.1156921, 16.1501732, -13.0678263, 16.0958939, -22.8793793, 22.8674088
17: -46.6164551, -12.0142307, -46.4013176, -12.0236931, -28.9358139, 28.6626587
18: -17.1616459, 11.4884138, -17.0915642, 11.4595757, -25.4483719, 25.4357681
19: -16.7745819, 6.1431427, -16.7315483, 6.1186943, -17.0704041, 17.0898247
20: -19.5903492, 2.9143496, -19.5749416, 2.8832173, -17.3904495, 17.3992195
21: -19.4153347, 5.5102892, -19.3671951, 5.4649191, -19.1020126, 19.1278229
22: -19.9810104, 6.1152692, -19.9261131, 6.1000347, -18.8610382, 18.8278961
23: -15.8774948, 9.3097219, -15.8373203, 9.2887230, -19.5341339, 19.5149422
24: -14.6942749, 9.7162027, -14.6383848, 9.6957970, -20.6543198, 20.6162186
25: -19.5028400, 5.9102221, -19.4438286, 5.8761177, -20.3634949, 20.3224411
26: -33.1988564, 4.2967896, -33.1397858, 4.2768998, -27.4714355, 27.4355698
27: -16.4261894, 11.2671986, -16.3956547, 11.1805954, -27.6067848, 27.6628532
28: -18.5316505, 9.6509781, -18.4990711, 9.6021223, -22.1631546, 22.1751328
29: -17.4183769, 3.7213337, -17.3662510, 3.6731968, -17.5339584, 17.5148621
30: -24.8358536, 1.7995596, -24.7935867, 1.7450690, -24.1643066, 24.1248856
31: -17.8449059, 6.6499767, -17.7884998, 6.6235332, -18.7116089, 18.7224579
32: -24.1057682, 2.7748704, -24.0869751, 2.6548448, -23.1045914, 23.2258148
33: -40.9134827, -1.9610596, -40.8982544, -2.0322061, -32.1977386, 32.2472076
34: -40.1334305, -9.3923512, -40.0974350, -9.4906406, -24.6364365, 24.6943588
35: -30.2149696, 0.5094199, -30.1852303, 0.4502110, -24.5109253, 24.5485535
36: -30.1113434, 4.8683872, -30.0787086, 4.7438240, -28.9445190, 29.0455627
37: -47.5524673, -6.6946244, -47.5162506, -6.8185964, -33.8263855, 33.9201050
38: -38.1422043, 4.8558826, -38.1015930, 4.7272019, -37.6003418, 37.6991730
39: -42.1622810, -1.2671194, -42.1432114, -1.3403721, -39.2956543, 39.3833618
40: -39.7357712, -10.8270912, -39.7218552, -10.9680862, -24.7601700, 24.8896713
41: -22.8865547, 6.4029207, -22.8613091, 6.2589903, -24.5830536, 24.7187500
42: -24.6530800, -2.1232121, -24.6375580, -2.2467144, -15.8255692, 15.9430809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1534486, upper bound: 9.2020791
time: 31.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1727986, upper bound: 9.2029992
time: 24.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.5296783, 17.7739563, -7.5370216, 17.7629566, -21.9764862, 22.0253143
1: -0.8619301, 19.0453091, -0.8690960, 19.0403442, -14.2159424, 14.2460327
2: -1.3765483, 19.0083733, -1.3378663, 19.0082417, -15.4908218, 15.4511642
3: -7.6549215, 15.5592384, -7.6336303, 15.5579119, -16.7626419, 16.7532806
4: -5.0699697, 17.5736542, -5.0489321, 17.5729561, -18.4840851, 18.5014992
5: -6.6083221, 18.0323563, -6.5989547, 18.0354366, -18.5601349, 18.5742645
6: -32.9786377, -5.4049034, -32.9698143, -5.5147018, -21.0775528, 21.1836548
7: -5.8214569, 18.0779266, -5.8155551, 18.0765305, -16.4393463, 16.4627914
8: -16.6373634, 8.8584108, -16.6320343, 8.8561192, -20.7385712, 20.7540855
9: -5.6877866, 19.6920223, -5.7236285, 19.6484833, -21.6166763, 21.6643066
10: -21.9651527, 13.0373077, -21.9202919, 13.0238304, -27.5430069, 27.4387054
11: -16.7287521, 6.0177898, -16.6973763, 6.0405397, -18.2145500, 18.1371078
12: -35.2597351, -3.5126433, -35.2518387, -3.5244293, -26.6447067, 26.6127243
13: -21.3654785, 16.3037701, -21.3374233, 16.2989807, -34.9722137, 34.9444275
14: -52.9178429, -14.7376213, -52.8009491, -14.7233896, -28.0470810, 27.8948135
15: -13.1097679, 11.8813887, -13.1092815, 11.8794422, -18.8857155, 18.8847389
16: -13.1199217, 16.1510830, -13.1111212, 16.1074524, -22.9177094, 22.8993301
17: -46.6268692, -12.0127668, -46.4568558, -11.9445782, -29.0357208, 28.7328339
18: -17.1777077, 11.4889526, -17.1534958, 11.5453205, -25.5499954, 25.4938965
19: -16.7814331, 6.1433411, -16.7651939, 6.1481433, -17.1069641, 17.1210747
20: -19.5929337, 2.9151063, -19.5928230, 2.9258492, -17.4210091, 17.4146080
21: -19.4240990, 5.5106573, -19.4045868, 5.5197496, -19.1699524, 19.1625481
22: -19.9863739, 6.1154661, -19.9518585, 6.1208363, -18.8889542, 18.8539467
23: -15.8874979, 9.3107138, -15.8769455, 9.3505869, -19.6052551, 19.5530396
24: -14.7033043, 9.7167578, -14.6822128, 9.7648287, -20.7331543, 20.6580811
25: -19.5111771, 5.9112005, -19.4849281, 5.9592867, -20.4543915, 20.3633804
26: -33.2147484, 4.2971911, -33.2005539, 4.3644686, -27.5742645, 27.4892578
27: -16.4290504, 11.2680511, -16.4185028, 11.2037678, -27.6328182, 27.6865540
28: -18.5423355, 9.6515093, -18.5418015, 9.6716747, -22.2431946, 22.2145157
29: -17.4254131, 3.7216029, -17.3968086, 3.6908526, -17.5672913, 17.5463791
30: -24.8504944, 1.8009806, -24.8504105, 1.8603058, -24.2969818, 24.1789627
31: -17.8496933, 6.6507998, -17.8142033, 6.6499634, -18.7447052, 18.7502403
32: -24.1070938, 2.7765961, -24.1285515, 2.6913171, -23.1374817, 23.2890549
33: -40.9144363, -1.9583063, -40.9200935, -2.0025010, -32.2266998, 32.2760010
34: -40.1451035, -9.3898077, -40.1443405, -9.4135151, -24.7253418, 24.7387848
35: -30.2236633, 0.5114956, -30.2258930, 0.5074739, -24.5770645, 24.5887680
36: -30.1183586, 4.8699570, -30.1064911, 4.7874084, -28.9985504, 29.0762482
37: -47.5634117, -6.6934395, -47.5663757, -6.7581329, -33.9005585, 33.9710159
38: -38.1520576, 4.8590412, -38.1413498, 4.8065186, -37.6944580, 37.7433777
39: -42.1627731, -1.2632952, -42.1624298, -1.3174577, -39.3205261, 39.4075012
40: -39.7391586, -10.8240032, -39.7460747, -10.9442654, -24.7935333, 24.9207230
41: -22.8893452, 6.4036222, -22.8829346, 6.2801218, -24.6158295, 24.7480469
42: -24.6524830, -2.1220257, -24.6425896, -2.2169743, -15.8522224, 15.9642906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=194, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=155, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 811

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1836187, upper bound: 9.2020791
time: 43.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2029990, upper bound: 9.2029992
time: 33.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 78.39 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.1836187, upper bound: 9.1810043
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.2029990, upper bound: 9.1818835
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.1534486, upper bound: 9.2020791
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.1727986, upper bound: 9.2029992
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.1836187, upper bound: 9.2020791
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 78.39
Output dim: 2, lower bound: -9.2029990, upper bound: 9.2029992

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.4409361, 17.7579727, -7.5346851, 17.7625217, -21.8894043, 21.9204407
1: -0.8088772, 19.0349483, -0.8667710, 19.0400810, -14.1321754, 14.1772499
2: -1.2573218, 18.9791546, -1.3364489, 18.9993248, -15.3370209, 15.4222870
3: -7.5729346, 15.5496264, -7.6315794, 15.5574455, -16.6600723, 16.7749710
4: -4.9445820, 17.5573864, -5.0460434, 17.5685120, -18.3573189, 18.4706421
5: -6.5280309, 18.0262337, -6.5974340, 18.0364838, -18.4589157, 18.5647392
6: -32.9213600, -5.5588069, -32.9534531, -5.5154209, -21.0143433, 20.8957405
7: -5.7521682, 18.0703754, -5.8126526, 18.0758553, -16.3229790, 16.3256302
8: -16.5248451, 8.8324146, -16.6294708, 8.8506355, -20.5918961, 20.7101364
9: -5.6599903, 19.6452370, -5.7267451, 19.6483822, -21.5555878, 21.6928406
10: -21.8752995, 12.9997005, -21.9170971, 13.0211697, -27.3639297, 27.4084625
11: -16.6844101, 5.9496169, -16.6964722, 6.0375090, -18.1536446, 18.0838165
12: -35.2370529, -3.6137600, -35.2477837, -3.5281010, -26.5731049, 26.5687256
13: -21.3015652, 16.2801361, -21.3346939, 16.2978859, -34.8980713, 34.9105988
14: -52.7732697, -14.8004093, -52.8001862, -14.7429905, -27.5310135, 27.8602295
15: -13.0176277, 11.8743973, -13.1067209, 11.8787899, -18.7271271, 18.8741226
16: -13.0788898, 16.0971680, -13.1105661, 16.1070271, -22.8325806, 22.9161301
17: -46.4354782, -12.0770435, -46.4555740, -11.9661007, -28.4907608, 28.7083435
18: -17.1387405, 11.4547567, -17.1534729, 11.5443382, -25.6421051, 25.4217834
19: -16.7509117, 6.1158199, -16.7641392, 6.1464958, -17.0934219, 17.0239792
20: -19.5861225, 2.8825819, -19.5928001, 2.9240739, -17.3901329, 17.3553314
21: -19.3940506, 5.4604378, -19.4035873, 5.5171504, -19.1413498, 19.0704803
22: -19.9386559, 6.0963097, -19.9496193, 6.1187415, -18.7672195, 18.7816086
23: -15.8677435, 9.2855062, -15.8765125, 9.3480091, -19.5800171, 19.4642639
24: -14.6647663, 9.6869049, -14.6821747, 9.7638865, -20.7073288, 20.5928879
25: -19.4704723, 5.8667254, -19.4849987, 5.9564710, -20.3622742, 20.2675095
26: -33.1850548, 4.2728291, -33.1994324, 4.3623810, -27.5369797, 27.3996124
27: -16.3850632, 11.1782341, -16.4128590, 11.2016544, -27.5867176, 27.5910931
28: -18.5268745, 9.5981789, -18.5413723, 9.6691504, -22.1869812, 22.1019135
29: -17.3830204, 3.6702170, -17.3951759, 3.6892772, -17.4757538, 17.4582672
30: -24.8387604, 1.7435536, -24.8502750, 1.8579826, -24.2109528, 24.1274567
31: -17.8034172, 6.6159372, -17.8143826, 6.6474080, -18.7154083, 18.6699028
32: -24.0452347, 2.6580734, -24.1119270, 2.6910348, -23.0683594, 23.1366425
33: -40.9042015, -2.0289493, -40.9196510, -2.0053349, -32.1925201, 32.1998062
34: -40.1053810, -9.4869089, -40.1347580, -9.4161844, -24.6967773, 24.6420517
35: -30.2028732, 0.4532681, -30.2233639, 0.5046797, -24.5795746, 24.5041656
36: -30.0548859, 4.7426324, -30.0896931, 4.7842183, -28.9398346, 28.9218292
37: -47.5266228, -6.8253169, -47.5584908, -6.7629151, -33.8596191, 33.8108826
38: -38.0928497, 4.7323637, -38.1279526, 4.8036470, -37.6564331, 37.5698700
39: -42.1399612, -1.3289156, -42.1600761, -1.3183460, -39.3013000, 39.3106995
40: -39.7082291, -10.9580221, -39.7383575, -10.9448280, -24.7462158, 24.7567902
41: -22.8152390, 6.2559814, -22.8625107, 6.2789621, -24.5499420, 24.4682236
42: -24.6058197, -2.2472484, -24.6285324, -2.2182987, -15.8057060, 15.8199844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=153, inp2_unstable=155, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1983285, upper bound: 9.1583277
time: 29.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2022267, upper bound: 9.1811117
time: 30.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.5225868, 17.7686481, -7.4350662, 17.7441063, -21.9515991, 21.9158249
1: -0.8553922, 19.0395699, -0.8027003, 19.0190258, -14.1607323, 14.1735115
2: -1.3731692, 18.9988842, -1.2565162, 18.9748192, -15.4316368, 15.3618698
3: -7.6512728, 15.5555534, -7.5744610, 15.5409403, -16.7178307, 16.6878510
4: -5.0636945, 17.5645943, -4.9445934, 17.5428619, -18.4057655, 18.3889236
5: -6.6043191, 18.0254498, -6.5281167, 18.0086174, -18.4964867, 18.4915428
6: -32.9763832, -5.4069123, -32.9589958, -5.5602531, -21.0099792, 21.1585274
7: -5.8137817, 18.0714912, -5.7477369, 18.0536346, -16.3552666, 16.3890495
8: -16.6324558, 8.8455219, -16.5259018, 8.8085222, -20.6559525, 20.6359215
9: -5.6844311, 19.6864586, -5.6557198, 19.6303120, -21.5839539, 21.5890121
10: -21.9588470, 13.0322123, -21.8719254, 12.9986095, -27.5071869, 27.3831329
11: -16.7131100, 6.0128469, -16.6392174, 5.9523830, -18.1093216, 18.0683746
12: -35.2541618, -3.5212097, -35.2332573, -3.6113467, -26.5502472, 26.6066666
13: -21.3603039, 16.2987709, -21.3013954, 16.2683964, -34.9327850, 34.9039764
14: -52.9148636, -14.7417927, -52.7722435, -14.7537956, -28.0023041, 27.8779984
15: -13.1040688, 11.8719940, -13.0135708, 11.8490381, -18.8599091, 18.7742043
16: -13.1118546, 16.1500320, -13.0656662, 16.0958519, -22.8372345, 22.8663177
17: -46.6141815, -12.0171728, -46.4001541, -12.0251360, -28.9209137, 28.6815414
18: -17.1607056, 11.4853973, -17.0910587, 11.4581032, -25.4461288, 25.4126129
19: -16.7738419, 6.1399307, -16.7311783, 6.1171489, -17.0688782, 17.0440788
20: -19.5899906, 2.9116476, -19.5747414, 2.8819578, -17.3887672, 17.3675995
21: -19.4144077, 5.5070705, -19.3667374, 5.4633989, -19.1004486, 19.0969887
22: -19.9803677, 6.1115851, -19.9257889, 6.0982847, -18.8591461, 18.7825127
23: -15.8769770, 9.3056774, -15.8370571, 9.2867632, -19.5323792, 19.4516640
24: -14.6938972, 9.7127199, -14.6382608, 9.6941547, -20.6529846, 20.5799484
25: -19.5024681, 5.9059434, -19.4436531, 5.8740659, -20.3616409, 20.2668915
26: -33.1982765, 4.2936416, -33.1395035, 4.2754822, -27.4694824, 27.3735657
27: -16.4257050, 11.2643585, -16.3954372, 11.1789761, -27.6046810, 27.6597958
28: -18.5311203, 9.6464405, -18.4988194, 9.5999374, -22.1612778, 22.1132736
29: -17.4174614, 3.7178049, -17.3657532, 3.6715040, -17.5315781, 17.4746323
30: -24.8353558, 1.7968388, -24.7933273, 1.7437820, -24.1625519, 24.1101990
31: -17.8442192, 6.6466160, -17.7882004, 6.6219234, -18.7098923, 18.6954956
32: -24.1041698, 2.7738481, -24.0861759, 2.6543422, -23.0920181, 23.2230606
33: -40.9124222, -1.9650445, -40.8977585, -2.0341635, -32.1881409, 32.2477722
34: -40.1329498, -9.3985176, -40.0971527, -9.4936180, -24.6349030, 24.6912079
35: -30.2142143, 0.5047812, -30.1848316, 0.4475989, -24.5082703, 24.5284424
36: -30.1110077, 4.8619013, -30.0785789, 4.7407045, -28.9421844, 29.0326614
37: -47.5509796, -6.7036572, -47.5156174, -6.8228745, -33.8154449, 33.9152145
38: -38.1416359, 4.8496265, -38.1013145, 4.7242513, -37.5967407, 37.6702728
39: -42.1589966, -1.2684908, -42.1416893, -1.3410630, -39.2913361, 39.3809052
40: -39.7325821, -10.8283405, -39.7202988, -10.9686985, -24.7387543, 24.8867645
41: -22.8856926, 6.3993120, -22.8608532, 6.2572021, -24.5805817, 24.7139359
42: -24.6528168, -2.1266236, -24.6374207, -2.2484169, -15.8214569, 15.9418564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 811

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1682127, upper bound: 9.1793717
time: 26.01 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1720263, upper bound: 9.2022267
time: 26.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.5252991, 17.7738171, -7.5348625, 17.7628746, -21.9699402, 22.0226288
1: -0.8578324, 19.0451279, -0.8670917, 19.0402317, -14.1809540, 14.2443810
2: -1.3739340, 19.0082111, -1.3364761, 19.0081711, -15.4608040, 15.4502640
3: -7.6522446, 15.5589809, -7.6323495, 15.5578003, -16.7352180, 16.7491226
4: -5.0650091, 17.5734673, -5.0465283, 17.5728836, -18.4343491, 18.4992142
5: -6.6053391, 18.0320892, -6.5975409, 18.0352993, -18.5205421, 18.5685120
6: -32.9772568, -5.4058523, -32.9691620, -5.5151896, -21.0650482, 21.1756592
7: -5.8160782, 18.0777054, -5.8128204, 18.0764427, -16.3793411, 16.4606705
8: -16.6330299, 8.8579283, -16.6297646, 8.8559055, -20.7023201, 20.7525101
9: -5.6861014, 19.6918774, -5.7225790, 19.6483765, -21.6012497, 21.6628571
10: -21.9613743, 13.0367546, -21.9184151, 13.0235710, -27.5352249, 27.4363480
11: -16.7273979, 6.0136242, -16.6967125, 6.0385404, -18.2120781, 18.1227303
12: -35.2591324, -3.5198884, -35.2515526, -3.5279560, -26.6338043, 26.6336517
13: -21.3617401, 16.3032665, -21.3356075, 16.2987061, -34.9656982, 34.9427338
14: -52.9161758, -14.7400179, -52.8000908, -14.7246208, -28.0361176, 27.9115829
15: -13.1063099, 11.8805122, -13.1076326, 11.8790169, -18.8903198, 18.8805656
16: -13.1160879, 16.1509438, -13.1089678, 16.1073914, -22.8755417, 22.8982391
17: -46.6245346, -12.0156775, -46.4558220, -11.9460258, -29.0207901, 28.7517548
18: -17.1767693, 11.4859505, -17.1530685, 11.5438347, -25.5477219, 25.4707336
19: -16.7807121, 6.1401386, -16.7648754, 6.1466455, -17.1054459, 17.0753479
20: -19.5925407, 2.9124477, -19.5926285, 2.9245293, -17.4193192, 17.3830109
21: -19.4232025, 5.5074687, -19.4040947, 5.5182185, -19.1684494, 19.1316872
22: -19.9857101, 6.1117668, -19.9515629, 6.1190405, -18.8871002, 18.8085976
23: -15.8869495, 9.3066368, -15.8766785, 9.3485985, -19.6035233, 19.4897385
24: -14.7029524, 9.7132874, -14.6820745, 9.7631884, -20.7318268, 20.6217880
25: -19.5107918, 5.9069042, -19.4847641, 5.9572191, -20.4525833, 20.3078308
26: -33.2142601, 4.2940006, -33.2003059, 4.3628855, -27.5723267, 27.4273605
27: -16.4286232, 11.2652407, -16.4182873, 11.2021513, -27.6307755, 27.6835289
28: -18.5417919, 9.6470175, -18.5415535, 9.6694765, -22.2412643, 22.1526566
29: -17.4244709, 3.7180660, -17.3963737, 3.6891675, -17.5649109, 17.5061417
30: -24.8499718, 1.7983046, -24.8501225, 1.8589916, -24.2952118, 24.1642456
31: -17.8490047, 6.6474390, -17.8138580, 6.6483221, -18.7429962, 18.7232590
32: -24.1054726, 2.7755322, -24.1277122, 2.6908636, -23.1249237, 23.2862930
33: -40.9133835, -1.9622827, -40.9195633, -2.0043912, -32.2170715, 32.2765579
34: -40.1446266, -9.3960571, -40.1440353, -9.4165068, -24.7238159, 24.7355423
35: -30.2229309, 0.5068517, -30.2255516, 0.5048795, -24.5744171, 24.5686646
36: -30.1180725, 4.8633895, -30.1063271, 4.7842140, -28.9962463, 29.0633240
37: -47.5618439, -6.7023840, -47.5655365, -6.7624836, -33.8895264, 33.9660645
38: -38.1515121, 4.8527932, -38.1410751, 4.8034501, -37.6909637, 37.7144470
39: -42.1595078, -1.2647061, -42.1608505, -1.3181891, -39.3162384, 39.4051361
40: -39.7359810, -10.8253183, -39.7445068, -10.9448595, -24.7721405, 24.9178314
41: -22.8884182, 6.3999839, -22.8824749, 6.2784200, -24.6133728, 24.7432175
42: -24.6522141, -2.1254439, -24.6424675, -2.2186875, -15.8481178, 15.9630508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=155, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 811

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1983285, upper bound: 9.1793717
time: 26.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2022267, upper bound: 9.2022267
time: 66.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 95.14 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.1983285, upper bound: 9.1583277
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.2022267, upper bound: 9.1811117
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.1682127, upper bound: 9.1793717
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.1720263, upper bound: 9.2022267
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.1983285, upper bound: 9.1793717
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.14
Output dim: 2, lower bound: -9.2022267, upper bound: 9.2022267

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 43.52 + 575.07 = 618.59 seconds
