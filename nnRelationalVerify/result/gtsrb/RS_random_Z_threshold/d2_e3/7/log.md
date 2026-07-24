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
execution time: IAR + RelationalAnalysis = 2.37 + 41.35 = 43.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -9.2114816, upper bound: 9.2114816

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1288

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2047229, upper bound: 9.2110878
time: 31.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2110878, upper bound: 9.2047228
time: 36.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 68.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 68.28
Output dim: 2, lower bound: -9.2047229, upper bound: 9.2110878
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 68.28
Output dim: 2, lower bound: -9.2110878, upper bound: 9.2047228

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.9016418, 21.9031448
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1665115, 14.1696243
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3833313, 15.3834457
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6941452, 16.6938019
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3848419, 18.3865395
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.5053558, 18.5063553
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0342560, 21.0342522
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3829384, 16.3868256
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6304131, 20.6336823
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5787354, 21.5815582
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4135513, 27.4175110
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0870399, 18.0885162
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5502930, 26.5503082
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9033051, 34.9016113
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8668060, 27.8736038
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7993965, 18.7993164
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8491364, 22.8549805
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7476349, 28.7476578
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4574051, 25.4577713
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0739670, 17.0740128
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3904800, 17.3897438
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1003113, 19.1019516
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8296890, 18.8281937
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5331345, 19.5320702
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6370850, 20.6370468
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3370285, 20.3366508
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4676590, 27.4647751
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1696320, 22.1672974
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5084000, 17.5076294
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1611099, 24.1618729
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7006226, 18.7022934
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1175003, 23.1163788
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1987000, 32.1939240
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6604843, 24.6568298
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5231400, 24.5181656
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9544983, 28.9505157
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8512268, 33.8492203
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6194153, 37.6159515
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3211670, 39.3210754
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7964020, 24.7965775
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5904770, 24.5892868
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8305855, 15.8290520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 861

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1422

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2027435, upper bound: 9.2029104
time: 31.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1965453, upper bound: 9.2091076
time: 26.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.9031448, 21.9016418
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1696243, 14.1665115
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3834457, 15.3833351
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6938019, 16.6941452
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3865433, 18.3848381
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.5063553, 18.5053520
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0342560, 21.0342522
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3868217, 16.3829346
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6336784, 20.6304169
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5815582, 21.5787354
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4175034, 27.4135590
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0885201, 18.0870361
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5503082, 26.5502930
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9016266, 34.9033051
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8736115, 27.8667984
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7993126, 18.7993927
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8549805, 22.8491364
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7476654, 28.7476273
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4577713, 25.4574051
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0740128, 17.0739632
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3897476, 17.3904762
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1019516, 19.1003113
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8281937, 18.8296890
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5320740, 19.5331345
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6370468, 20.6370850
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3366470, 20.3370285
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4647751, 27.4676590
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1672897, 22.1696320
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5076294, 17.5084000
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1618729, 24.1611099
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7022934, 18.7006264
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1163788, 23.1175003
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1939240, 32.1987000
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6568222, 24.6604919
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5181656, 24.5231400
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9505157, 28.9544983
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8492126, 33.8512344
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6159515, 37.6194153
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3210754, 39.3211823
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7965698, 24.7964020
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5892868, 24.5904694
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8290520, 15.8305855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1617

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1415

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1996743, upper bound: 9.2011852
time: 28.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2075549, upper bound: 9.1933017
time: 26.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 57.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 57.39
Output dim: 2, lower bound: -9.2027435, upper bound: 9.2029104
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 57.39
Output dim: 2, lower bound: -9.1965453, upper bound: 9.2091076
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 57.39
Output dim: 2, lower bound: -9.1996743, upper bound: 9.2011852
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 57.39
Output dim: 2, lower bound: -9.2075549, upper bound: 9.1933017

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8986893, 21.8994598
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1621895, 14.1641426
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3806572, 15.3800049
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6877060, 16.6856728
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3803978, 18.3809204
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4996109, 18.4992371
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0340347, 21.0340576
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3790169, 16.3817902
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6268692, 20.6291275
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5758133, 21.5778198
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4133224, 27.4172745
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0825233, 18.0849953
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5455551, 26.5466080
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9032288, 34.9015350
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8663559, 27.8735199
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7952271, 18.7938652
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8476639, 22.8530884
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7404785, 28.7420502
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4567261, 25.4570465
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0660477, 17.0676270
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3857346, 17.3861008
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0969391, 19.0992661
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8260880, 18.8253479
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5272064, 19.5274048
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6332016, 20.6340179
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3312836, 20.3322220
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4622116, 27.4604950
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1639175, 22.1630630
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5042877, 17.5043297
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1582260, 24.1598816
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6951752, 18.6980362
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1174011, 23.1162949
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1963043, 32.1923065
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6603394, 24.6567535
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5208664, 24.5163651
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9524689, 28.9489288
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8441162, 33.8438416
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6170959, 37.6141357
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3173218, 39.3180084
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7957153, 24.7965469
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5892410, 24.5883865
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8296471, 15.8289146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 981

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 837

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1825481, upper bound: 9.2027949
time: 22.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2026279, upper bound: 9.1827119
time: 28.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8979568, 21.9001923
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1610298, 14.1653023
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3798943, 15.3807678
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6860199, 16.6873627
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3792152, 18.3821030
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4982376, 18.5006142
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0340576, 21.0340347
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3778954, 16.3829079
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6258621, 20.6301384
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5750046, 21.5786362
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4133224, 27.4172745
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0835152, 18.0839996
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5465927, 26.5455704
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9032135, 34.9015503
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8667221, 27.8731537
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7939453, 18.7951508
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8472443, 22.8535080
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7420197, 28.7405090
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4566803, 25.4570847
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0675735, 17.0660934
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3868332, 17.3850021
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0976257, 19.0985718
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8268509, 18.8245850
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5284653, 19.5261383
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6340561, 20.6331558
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3326035, 20.3309021
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4633713, 27.4593201
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1653976, 22.1615829
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5050964, 17.5035172
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1591187, 24.1589890
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6963654, 18.6968460
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1174164, 23.1162796
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1970825, 32.1915283
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6604004, 24.6566772
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5213394, 24.5158997
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9529266, 28.9484863
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8458710, 33.8421021
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6176147, 37.6136169
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3181152, 39.3171844
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7963715, 24.7958832
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5895767, 24.5880508
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8304482, 15.8281136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1288

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1884784, upper bound: 9.2082229
time: 34.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1956403, upper bound: 9.2010451
time: 32.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.9031448, 21.9006271
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1696243, 14.1648636
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3834457, 15.3828964
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6936913, 16.6941452
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3865433, 18.3833504
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.5063553, 18.5052185
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0341110, 21.0342522
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3868217, 16.3804703
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6336784, 20.6294708
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5811157, 21.5787354
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4161377, 27.4135590
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0885201, 18.0868988
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5485840, 26.5502930
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9003754, 34.9033051
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8726044, 27.8667984
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7984009, 18.7993927
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8549805, 22.8481369
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7476654, 28.7476120
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4577713, 25.4564667
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0740128, 17.0738297
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3893509, 17.3904762
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1019211, 19.1003113
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8273392, 18.8296890
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5318222, 19.5331345
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6370468, 20.6363907
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3366470, 20.3367844
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4625931, 27.4676590
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1672440, 22.1696320
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5063858, 17.5084000
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1618729, 24.1609268
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7022934, 18.6993713
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1156387, 23.1175003
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1935577, 32.1987000
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6563339, 24.6604919
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5175095, 24.5231400
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9497681, 28.9544983
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8488464, 33.8512344
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6156158, 37.6194153
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3208923, 39.3211823
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7965698, 24.7961502
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5889587, 24.5904694
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8281250, 15.8305855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1349

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2058604, upper bound: 9.1815705
time: 18.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1958391, upper bound: 9.1915824
time: 27.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 48.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.1825481, upper bound: 9.2027949
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.2026279, upper bound: 9.1827119
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.1884784, upper bound: 9.2082229
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.1956403, upper bound: 9.2010451
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.2058604, upper bound: 9.1815705
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 48.01
Output dim: 2, lower bound: -9.1958391, upper bound: 9.1915824

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8974380, 21.8990097
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1592331, 14.1618538
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3668518, 15.3687630
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6864281, 16.6848907
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3723450, 18.3752327
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4985580, 18.4987106
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0142746, 21.0103874
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3783264, 16.3818970
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6215286, 20.6248932
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5764999, 21.5774841
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4222717, 27.4232254
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0804138, 18.0823441
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5363159, 26.5343857
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9034119, 34.9017029
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8592300, 27.8675842
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7892532, 18.7880173
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8454819, 22.8489914
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7348938, 28.7375488
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4535904, 25.4559937
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0667572, 17.0685921
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3858032, 17.3858986
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0954819, 19.0977097
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8260574, 18.8254585
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5266800, 19.5270233
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6264191, 20.6285858
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3328781, 20.3338203
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4586258, 27.4583435
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1639099, 22.1631927
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5036163, 17.5036583
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1601486, 24.1597366
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6952057, 18.6984406
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1063538, 23.1029968
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1939697, 32.1874771
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6439285, 24.6368256
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5152664, 24.5101700
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9497833, 28.9459915
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8313904, 33.8275452
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6158295, 37.6127319
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3163757, 39.3163300
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7799759, 24.7770920
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5769348, 24.5736847
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8140602, 15.8094330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1339

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1438

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1746974, upper bound: 9.1937491
time: 27.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1720116, upper bound: 9.1952219
time: 24.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8982391, 21.8982010
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1598969, 14.1611900
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3694153, 15.3661995
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6869240, 16.6843948
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3747101, 18.3728638
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4990845, 18.4981842
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0103683, 21.0142937
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3791275, 16.3810997
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6226425, 20.6237793
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5754776, 21.5785065
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4192657, 27.4262161
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0798721, 18.0828857
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5333328, 26.5373688
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9033966, 34.9017334
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8604050, 27.8664017
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7893829, 18.7878876
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8435669, 22.8509140
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7359772, 28.7364731
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4556808, 25.4539032
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0670090, 17.0683441
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3855362, 17.3861694
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0953827, 19.0978088
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8261948, 18.8253288
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5268173, 19.5268784
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6277695, 20.6272354
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3328857, 20.3338165
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4600601, 27.4569168
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1640472, 22.1630554
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5036163, 17.5036621
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1580734, 24.1618118
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6955795, 18.6980667
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1041107, 23.1052399
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1914673, 32.1899796
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6404190, 24.6403275
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5146713, 24.5107651
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9495392, 28.9462357
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8278198, 33.8311157
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6156769, 37.6128693
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3156433, 39.3170929
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7762527, 24.7808228
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5745392, 24.5760880
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8101654, 15.8133278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 992

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1509

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2009257, upper bound: 9.1823350
time: 26.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2022493, upper bound: 9.1810157
time: 20.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8957138, 21.8984909
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1580467, 14.1630363
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3786507, 15.3798676
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6842422, 16.6860352
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3773155, 18.3806610
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4963379, 18.4991913
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0339813, 21.0339737
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3760414, 16.3815689
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6240311, 20.6287575
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5744095, 21.5781326
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4129562, 27.4169922
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0825310, 18.0826569
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5463181, 26.5453262
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9022827, 34.9008942
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8643112, 27.8716202
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7920380, 18.7937508
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8447571, 22.8517532
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7415466, 28.7402191
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4565201, 25.4570236
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0649796, 17.0626640
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3839493, 17.3811989
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0954742, 19.0957336
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8248978, 18.8218765
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5260162, 19.5229111
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6323929, 20.6309662
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3301697, 20.3276787
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4609451, 27.4558716
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1627808, 22.1581345
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5039139, 17.5019569
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1576843, 24.1571045
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6941605, 18.6939430
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1170197, 23.1157684
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1949615, 32.1887360
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6601868, 24.6564941
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5211029, 24.5156021
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9528961, 28.9484558
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8452454, 33.8412018
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6175690, 37.6135864
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3169861, 39.3155975
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7961884, 24.7957916
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5895309, 24.5879822
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8292656, 15.8264847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1291

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1414

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1871383, upper bound: 9.2076505
time: 34.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1879071, upper bound: 9.2068768
time: 26.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8969498, 21.8935699
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1635971, 14.1581116
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3750000, 15.3734627
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6913490, 16.6911545
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3835487, 18.3798943
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.5048294, 18.5031891
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0264587, 21.0275726
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3836517, 16.3765068
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6257668, 20.6205368
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5804520, 21.5785751
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4148102, 27.4137344
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0884056, 18.0869370
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5385971, 26.5418167
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9008179, 34.9037018
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8699036, 27.8637161
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7981567, 18.7991409
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8525085, 22.8461685
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7439651, 28.7434387
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4569397, 25.4554901
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0747604, 17.0741425
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3868332, 17.3882980
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1015472, 19.1000900
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8263321, 18.8287621
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5315018, 19.5326614
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6331940, 20.6320343
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3362274, 20.3364105
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4610901, 27.4660873
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1673660, 22.1696548
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5067291, 17.5084648
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1600266, 24.1593628
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7024155, 18.6989822
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1046982, 23.1078339
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1910706, 32.1966858
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6385269, 24.6444244
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5140228, 24.5196762
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9436951, 28.9492264
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8371887, 33.8407898
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6103363, 37.6148529
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3165588, 39.3175964
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7834167, 24.7842255
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5824661, 24.5847473
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8216248, 15.8248100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1339

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1959383, upper bound: 9.1714680
time: 36.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1957520, upper bound: 9.1716545
time: 26.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 64.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1746974, upper bound: 9.1937491
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1720116, upper bound: 9.1952219
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.2009257, upper bound: 9.1823350
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.2022493, upper bound: 9.1810157
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1871383, upper bound: 9.2076505
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1879071, upper bound: 9.2068768
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1959383, upper bound: 9.1714680
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 64.87
Output dim: 2, lower bound: -9.1957520, upper bound: 9.1716545

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8951645, 21.8980255
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1570168, 14.1621552
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3773994, 15.3788071
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6823959, 16.6843491
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3704224, 18.3742867
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4962845, 18.4991341
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0359344, 21.0354004
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3774338, 16.3826332
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6234093, 20.6280479
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5671082, 21.5700455
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4009323, 27.4032898
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0726089, 18.0708199
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5431213, 26.5418701
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9014740, 34.9003143
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8575363, 27.8635101
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7854156, 18.7882538
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8309250, 22.8358002
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7427216, 28.7418976
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4551544, 25.4557419
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0675430, 17.0642624
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3790703, 17.3753242
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0946808, 19.0932274
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8184738, 18.8160744
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5250702, 19.5210037
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6314392, 20.6299896
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3284988, 20.3258133
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4605408, 27.4553909
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1630707, 22.1584167
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5019989, 17.5004578
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1551514, 24.1540909
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6946945, 18.6932869
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1164474, 23.1147690
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1872406, 32.1822357
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6504288, 24.6479797
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5119629, 24.5076065
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9486389, 28.9447556
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8443146, 33.8403625
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6176605, 37.6137543
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3178406, 39.3160400
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7966919, 24.7962341
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5903320, 24.5887527
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8196259, 15.8161621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1500

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1830166, upper bound: 9.2035384
time: 28.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1830166, upper bound: 9.2035384
time: 28.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8952560, 21.8979340
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1571617, 14.1620102
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3775902, 15.3786201
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6825638, 16.6841888
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3709412, 18.3737602
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4962845, 18.4991379
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0354080, 21.0359268
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3771057, 16.3829575
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6233330, 20.6281281
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5663223, 21.5708389
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3992538, 27.4049606
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0706863, 18.0727348
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5428619, 26.5421295
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9017181, 34.9000702
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8561935, 27.8648453
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7865524, 18.7871246
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8287964, 22.8379288
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7432251, 28.7413864
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4552460, 25.4556580
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0665741, 17.0652275
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3780785, 17.3763161
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0929565, 19.0949440
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8190918, 18.8154564
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5241089, 19.5219650
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6314163, 20.6300125
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3283081, 20.3260078
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4604645, 27.4554749
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1630630, 22.1584320
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5024185, 17.5000381
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1546783, 24.1545639
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6935196, 18.6944695
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1160126, 23.1151962
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1884460, 32.1810303
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6516647, 24.6467438
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5131073, 24.5064697
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9491882, 28.9441910
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8444061, 33.8402710
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6177368, 37.6136780
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3174133, 39.3164825
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7966309, 24.7963028
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5903015, 24.5887833
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8189468, 15.8168411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1619

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1871726, upper bound: 9.1910894
time: 38.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1721156, upper bound: 9.2061356
time: 30.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 71.46 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 71.46
Output dim: 2, lower bound: -9.1830166, upper bound: 9.2035384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 71.46
Output dim: 2, lower bound: -9.1830166, upper bound: 9.2035384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 71.46
Output dim: 2, lower bound: -9.1871726, upper bound: 9.1910894
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 71.46
Output dim: 2, lower bound: -9.1721156, upper bound: 9.2061356

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8959732, 21.8977127
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1568794, 14.1620789
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3773308, 15.3787079
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6821289, 16.6846542
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3702393, 18.3742180
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4960709, 18.4994049
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0344238, 21.0356560
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3764877, 16.3827744
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6230431, 20.6283531
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5675430, 21.5698013
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4009476, 27.4030380
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0717850, 18.0710945
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5426178, 26.5418396
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9015198, 34.9002228
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8577652, 27.8634338
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7864113, 18.7872429
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8301773, 22.8358688
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7426147, 28.7416840
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4547501, 25.4554520
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0678101, 17.0638504
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3789177, 17.3755798
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0946198, 19.0931740
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8186417, 18.8147240
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5251999, 19.5203705
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6317139, 20.6295776
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3286209, 20.3250656
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4604340, 27.4552460
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1632080, 22.1577988
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5022659, 17.4992981
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1551285, 24.1540909
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6946869, 18.6931000
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1155624, 23.1149368
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1874847, 32.1818542
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6503067, 24.6480865
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5120697, 24.5074463
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9481354, 28.9450226
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8442841, 33.8403625
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6166992, 37.6140747
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3177185, 39.3161469
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7954254, 24.7965393
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5897827, 24.5889587
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8192101, 15.8166122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1650

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1529

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1816060, upper bound: 9.1939600
time: 23.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1734401, upper bound: 9.2021287
time: 29.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8948441, 21.8980255
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1570168, 14.1620178
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3773994, 15.3787308
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6823959, 16.6840820
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3704224, 18.3741112
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4962845, 18.4989166
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0359344, 21.0338936
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3774338, 16.3816910
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6234093, 20.6276817
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5668640, 21.5700455
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4006729, 27.4032898
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0726089, 18.0699959
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5431213, 26.5413589
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.9013672, 34.9003143
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8574600, 27.8635101
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7844048, 18.7882538
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8309250, 22.8350525
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7427216, 28.7417908
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4551544, 25.4553452
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0671310, 17.0642624
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3790703, 17.3751717
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0946198, 19.0932274
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8171234, 18.8160744
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5244370, 19.5210037
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6310272, 20.6299896
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3277512, 20.3258133
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4605408, 27.4552917
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1624603, 22.1584167
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5008392, 17.5004578
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1551514, 24.1540680
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6945038, 18.6932869
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1164474, 23.1138840
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1868591, 32.1822357
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6504288, 24.6478653
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5117950, 24.5076065
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9486389, 28.9442596
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8443146, 33.8403320
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6176605, 37.6127930
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3178406, 39.3159180
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7966919, 24.7949677
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5903320, 24.5881958
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8196259, 15.8157425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1608

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1793576, upper bound: 9.1998948
time: 26.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1793663, upper bound: 9.1998860
time: 28.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8857727, 21.8903351
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1558037, 14.1607857
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3605995, 15.3673172
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6777534, 16.6833725
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3666916, 18.3709717
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4872742, 18.4956970
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0314789, 21.0309029
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3770027, 16.3824196
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6220093, 20.6281471
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5660553, 21.5705719
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3989487, 27.4047775
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0596848, 18.0566597
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5430450, 26.5419617
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8755493, 34.8820190
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8452530, 27.8572388
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7711220, 18.7777519
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8266296, 22.8361664
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7270660, 28.7284775
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4507294, 25.4493561
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0587311, 17.0541077
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3620567, 17.3524933
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0780182, 19.0731621
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8131256, 18.8066216
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5166245, 19.5110588
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6215668, 20.6158371
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3191071, 20.3104706
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4542694, 27.4479065
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1509857, 22.1405411
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5022888, 17.4999466
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1340790, 24.1243896
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6835556, 18.6800957
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1136322, 23.1115875
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1891327, 32.1778107
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6474228, 24.6414413
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5102081, 24.5072021
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9436646, 28.9434433
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8437653, 33.8395844
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6156921, 37.6126404
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3143311, 39.3136292
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7924652, 24.7898636
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5898743, 24.5889816
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8139572, 15.8077164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1617

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1709817, upper bound: 9.2061282
time: 25.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1721082, upper bound: 9.2050028
time: 32.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 60.35 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1816060, upper bound: 9.1939600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1734401, upper bound: 9.2021287
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1793576, upper bound: 9.1998948
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1793663, upper bound: 9.1998860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1709817, upper bound: 9.2061282
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 60.35
Output dim: 2, lower bound: -9.1721082, upper bound: 9.2050028

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8851471, 21.8898239
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1529160, 14.1583786
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3591614, 15.3660698
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6756287, 16.6815109
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3665543, 18.3711052
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4861526, 18.4947510
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0322876, 21.0315628
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3722420, 16.3782005
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6211090, 20.6277657
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5660400, 21.5705566
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3978043, 27.4037857
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0626030, 18.0593376
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5428848, 26.5418243
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8757324, 34.8822937
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8423920, 27.8548431
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7714310, 18.7782097
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8228989, 22.8330231
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7269974, 28.7284164
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4503403, 25.4490280
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0569458, 17.0520058
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3602219, 17.3502045
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0771255, 19.0720482
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8095245, 18.8022957
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5144882, 19.5084915
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6203995, 20.6144333
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3147507, 20.3054771
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4518738, 27.4449234
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1487961, 22.1379242
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5007553, 17.4981079
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1344376, 24.1246643
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6820374, 18.6783066
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1136322, 23.1115875
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1859131, 32.1739502
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6470184, 24.6410980
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5089264, 24.5060883
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9425659, 28.9424515
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8421936, 33.8378296
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6145630, 37.6116943
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3129883, 39.3120270
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7938614, 24.7913742
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5903778, 24.5894470
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8131943, 15.8067932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 992

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1441

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1708765, upper bound: 9.2058893
time: 29.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1707422, upper bound: 9.2060229
time: 24.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8852615, 21.8897095
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1533966, 14.1578979
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3593521, 15.3658752
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6758881, 16.6812477
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3668213, 18.3708344
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4863281, 18.4945755
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0321503, 21.0317154
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3727837, 16.3776550
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6216278, 20.6272507
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5660400, 21.5705566
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3979568, 27.4036407
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0623589, 18.0595856
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5429153, 26.5418015
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8758087, 34.8822021
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8428497, 27.8543777
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7715836, 18.7780571
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8234863, 22.8324432
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7270126, 28.7284012
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4504089, 25.4489594
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0566330, 17.0523224
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3597641, 17.3506584
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0769043, 19.0722694
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8088074, 18.8030205
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5140610, 19.5089226
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6201630, 20.6146698
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3141174, 20.3061142
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4512939, 27.4455032
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1483688, 22.1383514
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5004501, 17.4984131
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1343536, 24.1247482
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6817627, 18.6785774
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1136322, 23.1115952
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1852722, 32.1745911
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6470795, 24.6410294
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5090790, 24.5059280
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9426575, 28.9423599
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8420258, 33.8380127
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6147461, 37.6115112
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3127136, 39.3122864
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7939835, 24.7912521
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5903473, 24.5894775
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8130341, 15.8069496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1518

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 837

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1519086, upper bound: 9.2048872
time: 21.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1719928, upper bound: 9.1848177
time: 29.62 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 52.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 52.62
Output dim: 2, lower bound: -9.1708765, upper bound: 9.2058893
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 52.62
Output dim: 2, lower bound: -9.1707422, upper bound: 9.2060229
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 52.62
Output dim: 2, lower bound: -9.1519086, upper bound: 9.2048872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 52.62
Output dim: 2, lower bound: -9.1719928, upper bound: 9.1848177

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8846130, 21.8892746
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1523323, 14.1576767
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3580818, 15.3649292
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6772308, 16.6826706
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3649445, 18.3694229
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4858017, 18.4943733
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0317307, 21.0309639
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3710251, 16.3767738
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6209259, 20.6275864
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5657272, 21.5701294
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3976898, 27.4032593
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0618057, 18.0585251
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5441895, 26.5433884
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8753967, 34.8820343
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8349686, 27.8480377
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7705231, 18.7774010
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8228226, 22.8327103
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7234344, 28.7254486
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4506531, 25.4494705
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0560074, 17.0512314
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3602448, 17.3502388
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0772781, 19.0722580
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8090591, 18.8019333
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5134277, 19.5075417
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6203232, 20.6143646
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3140488, 20.3048630
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4504547, 27.4437866
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1486511, 22.1377869
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4992294, 17.4968300
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1344452, 24.1246872
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6820602, 18.6784286
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1131287, 23.1110077
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1831665, 32.1710358
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6472321, 24.6410370
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5089569, 24.5060577
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9418945, 28.9419327
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8412476, 33.8372345
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6146545, 37.6117859
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3131256, 39.3121796
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7938538, 24.7913666
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5902786, 24.5893250
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8136673, 15.8071747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1529

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1694755, upper bound: 9.1963303
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1613062, upper bound: 9.2044804
time: 30.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8845978, 21.8892899
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1522179, 14.1577911
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3580208, 15.3649902
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6767960, 16.6831055
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3648758, 18.3694878
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4857788, 18.4943924
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0316849, 21.0310020
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3708115, 16.3769836
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6209335, 20.6275787
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5656128, 21.5702362
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3972778, 27.4036713
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0617905, 18.0585327
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5444489, 26.5431290
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8754883, 34.8819427
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8355789, 27.8474274
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7706146, 18.7773094
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8225937, 22.8329468
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7240295, 28.7248611
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4507828, 25.4493408
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0561752, 17.0510674
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3602600, 17.3502274
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0773392, 19.0721970
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8091583, 18.8018265
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5135345, 19.5074310
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6203308, 20.6143570
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3141403, 20.3047676
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4507294, 27.4435120
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1486588, 22.1377792
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4994812, 17.4965820
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1344604, 24.1246796
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6821594, 18.6783333
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1130524, 23.1110840
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1830139, 32.1712036
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6469574, 24.6413040
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5088959, 24.5061264
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9420471, 28.9417801
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8415985, 33.8368835
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6146545, 37.6117859
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3131256, 39.3121643
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7938385, 24.7913742
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5902481, 24.5893478
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8135757, 15.8072662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1766

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1565365, upper bound: 9.2056925
time: 24.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1703779, upper bound: 9.1917946
time: 17.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8840027, 21.8892670
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1504364, 14.1556015
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3455429, 15.3546257
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6746216, 16.6804810
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3587723, 18.3651543
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4852600, 18.4940338
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0123672, 21.0080261
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3720970, 16.3777618
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6162567, 20.6230011
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5667191, 21.5702133
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4069214, 27.4096146
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0602493, 18.0569305
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5336685, 26.5295715
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8760071, 34.8823700
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8357544, 27.8484650
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7656021, 18.7722092
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8213043, 22.8283386
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7214127, 28.7238846
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4472656, 25.4479141
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0573425, 17.0532837
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3598175, 17.3504410
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0754547, 19.0707169
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8087769, 18.8031235
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5135345, 19.5085411
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6133957, 20.6092529
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3157043, 20.3077087
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4476929, 27.4433365
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1483459, 22.1384659
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4997635, 17.4977264
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1362915, 24.1246033
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6818008, 18.6789856
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1025772, 23.0982895
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1829529, 32.1697617
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6306610, 24.6211014
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5034943, 24.4997482
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9399414, 28.9393921
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8292694, 33.8216858
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6134949, 37.6101074
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3117981, 39.3106079
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7782288, 24.7717896
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5780334, 24.5747757
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7974358, 15.7874527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1621

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1516

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1504575, upper bound: 9.2034331
time: 25.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1504575, upper bound: 9.2034331
time: 26.29 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 54.06 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1694755, upper bound: 9.1963303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1613062, upper bound: 9.2044804
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1565365, upper bound: 9.2056925
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1703779, upper bound: 9.1917946
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1504575, upper bound: 9.2034331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 54.06
Output dim: 2, lower bound: -9.1504575, upper bound: 9.2034331

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8839569, 21.8889999
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1514893, 14.1570816
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3554878, 15.3628883
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6752396, 16.6811600
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3633041, 18.3681717
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4820938, 18.4914474
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0293350, 21.0292015
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3678818, 16.3742981
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6185417, 20.6257057
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5619202, 21.5670547
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3918762, 27.3983917
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0614929, 18.0580330
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5426559, 26.5411530
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8747101, 34.8817902
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8348694, 27.8478775
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7696533, 18.7767029
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8200378, 22.8310318
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7205887, 28.7217789
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4469986, 25.4448395
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0551147, 17.0502892
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3599701, 17.3492889
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0758438, 19.0701942
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8062134, 18.7983208
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5116425, 19.5050888
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6189804, 20.6123581
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3111877, 20.3008728
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4461670, 27.4383926
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1460800, 22.1343765
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4987717, 17.4964409
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1337204, 24.1235657
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6817551, 18.6781883
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1118393, 23.1095886
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1826477, 32.1697617
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6446686, 24.6378021
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5059052, 24.5021667
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9408417, 28.9406128
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8364258, 33.8313980
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6104736, 37.6067352
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3125305, 39.3117065
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7928772, 24.7903595
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5892944, 24.5883942
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8133659, 15.8068886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1388

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 843

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1488559, upper bound: 9.2042920
time: 19.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1611183, upper bound: 9.1920410
time: 25.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8793945, 21.8869553
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1478233, 14.1553116
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3544731, 15.3633881
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6761398, 16.6827469
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3591843, 18.3672409
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4822922, 18.4929352
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0274506, 21.0209236
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3669014, 16.3757477
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6156693, 20.6253586
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5598526, 21.5681152
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3923492, 27.4013672
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0616302, 18.0583382
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5452805, 26.5406570
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8748016, 34.8816986
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8251038, 27.8430328
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7670135, 18.7753906
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8167877, 22.8309021
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7245560, 28.7247620
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4500656, 25.4488373
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0560989, 17.0510254
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3583641, 17.3471336
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0761108, 19.0721970
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8085938, 18.8011017
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5134354, 19.5072899
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6201935, 20.6141052
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3139725, 20.3042221
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4490128, 27.4416809
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1464844, 22.1336670
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4989700, 17.4957771
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1333008, 24.1218109
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6816788, 18.6779976
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1098251, 23.1040268
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1790924, 32.1616058
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6418457, 24.6284561
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5058060, 24.4999619
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9391785, 28.9356308
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8390350, 33.8304749
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6117859, 37.6050262
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3125000, 39.3112640
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7901459, 24.7822418
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5873642, 24.5822296
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8098907, 15.7980156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1339

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1335

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1562143, upper bound: 9.2049530
time: 26.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1557994, upper bound: 9.2053706
time: 26.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8840103, 21.8891754
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1500931, 14.1555557
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3453751, 15.3546906
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6742439, 16.6806107
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3585281, 18.3653831
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4849396, 18.4941101
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0118027, 21.0077400
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3712807, 16.3776970
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6160049, 20.6232986
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5667877, 21.5701065
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4070435, 27.4094772
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0598526, 18.0566711
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5336761, 26.5294952
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8759003, 34.8822479
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8357086, 27.8483582
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7653809, 18.7717438
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8209534, 22.8282700
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7213821, 28.7237473
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4471512, 25.4481430
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0572968, 17.0528412
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3597946, 17.3503990
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0754395, 19.0706711
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8086929, 18.8023071
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5135193, 19.5081291
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6133881, 20.6090088
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3157959, 20.3072777
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4476929, 27.4433365
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1483154, 22.1380234
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4997482, 17.4969978
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1363144, 24.1245651
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6817703, 18.6788177
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1023483, 23.0982132
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1830444, 32.1691513
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6305466, 24.6208572
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5034943, 24.4993515
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9399261, 28.9393539
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8292389, 33.8212433
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6132812, 37.6101074
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3117218, 39.3104248
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7780304, 24.7718506
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5779724, 24.5747528
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7973595, 15.7874336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1381

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1353109, upper bound: 9.2027987
time: 27.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1498232, upper bound: 9.1882793
time: 29.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8839035, 21.8892670
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1504364, 14.1552582
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3455429, 15.3544617
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6746216, 16.6801033
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3587723, 18.3649178
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4852600, 18.4937134
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0123672, 21.0074654
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3720970, 16.3769417
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6162567, 20.6227455
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5666122, 21.5702133
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4067841, 27.4096146
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0602493, 18.0565414
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5335922, 26.5295715
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8758850, 34.8823700
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8356476, 27.8484650
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7651367, 18.7722092
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8213043, 22.8279877
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7212753, 28.7238846
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4472656, 25.4477921
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0569000, 17.0532837
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3598175, 17.3504219
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0754089, 19.0707169
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8079605, 18.8031235
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5131226, 19.5085411
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6131516, 20.6092529
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3152695, 20.3077087
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4476929, 27.4433365
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1479034, 22.1384659
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4990311, 17.4977264
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1362534, 24.1246033
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6816330, 18.6789856
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1025772, 23.0980606
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1823425, 32.1697617
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6304092, 24.6211014
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5030975, 24.4997482
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9399414, 28.9393539
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8288422, 33.8216858
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6134949, 37.6099091
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3115997, 39.3106079
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7782288, 24.7715759
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5780334, 24.5747070
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7974358, 15.7873802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1288

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1502660, upper bound: 9.2034306
time: 24.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1504550, upper bound: 9.2032420
time: 26.07 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 52.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1488559, upper bound: 9.2042920
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1611183, upper bound: 9.1920410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1562143, upper bound: 9.2049530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1557994, upper bound: 9.2053706
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1353109, upper bound: 9.2027987
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1498232, upper bound: 9.1882793
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1502660, upper bound: 9.2034306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 52.52
Output dim: 2, lower bound: -9.1504550, upper bound: 9.2032420

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8808746, 21.8871460
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1481552, 14.1550560
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3526268, 15.3615036
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6735611, 16.6798859
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3609161, 18.3677292
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4795761, 18.4900703
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0252533, 21.0238304
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3660393, 16.3733215
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6170616, 20.6248779
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5614090, 21.5663452
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3889618, 27.3933105
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0605507, 18.0553360
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5394516, 26.5362701
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8740845, 34.8813629
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8346024, 27.8477478
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7645874, 18.7736702
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8202744, 22.8306961
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7207870, 28.7217178
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4457016, 25.4447937
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0533676, 17.0471992
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3582573, 17.3464470
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0738831, 19.0667229
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8061752, 18.7982864
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5105820, 19.5029488
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6189728, 20.6123505
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3095779, 20.2981186
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4459229, 27.4382401
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1455917, 22.1328125
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4990768, 17.4964104
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1337967, 24.1218567
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6801071, 18.6754570
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1092453, 23.1059799
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1789856, 32.1638870
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6423569, 24.6333160
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5049896, 24.5000992
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9407654, 28.9405746
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8338623, 33.8266525
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6098938, 37.6062927
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3104401, 39.3079681
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7891693, 24.7850952
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5872498, 24.5855255
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8090782, 15.8001671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 827

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1432725, upper bound: 9.2042301
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1486982, upper bound: 9.1979307
time: 32.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8811722, 21.8882065
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1485367, 14.1559601
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3546219, 15.3630905
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6771660, 16.6841583
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3598785, 18.3681145
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4810219, 18.4921951
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0119171, 21.0076294
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3681564, 16.3773308
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6128769, 20.6220093
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5567627, 21.5656128
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3785095, 27.3898468
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0557709, 18.0515480
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5282593, 26.5254517
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8666534, 34.8747864
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8261948, 27.8439484
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7666206, 18.7750015
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8231659, 22.8396149
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7254562, 28.7254410
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4503403, 25.4490509
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0464020, 17.0396423
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3493347, 17.3369522
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0674133, 19.0619888
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8002090, 18.7913055
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5035477, 19.4957657
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6081009, 20.5999146
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3065033, 20.2954826
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4404373, 27.4316559
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1352692, 22.1206055
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4897156, 17.4850006
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1284180, 24.1160736
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6736221, 18.6685562
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0967789, 23.0928726
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1789398, 32.1614761
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6475830, 24.6370926
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4965591, 24.4916306
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9249268, 28.9231186
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8296814, 33.8230286
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6007843, 37.5954132
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3043671, 39.3043518
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7724228, 24.7669525
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5735016, 24.5701294
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8021240, 15.7915154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1338

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1546136, upper bound: 9.2033138
time: 27.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1546135, upper bound: 9.2033138
time: 32.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8806381, 21.8887253
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1484756, 14.1560211
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3541794, 15.3635406
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6775475, 16.6837769
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3600616, 18.3679390
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4815483, 18.4916725
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0141602, 21.0053864
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3684845, 16.3770065
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6123199, 20.6225700
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5573425, 21.5650330
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3808441, 27.3875122
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0548477, 18.0524750
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5300751, 26.5236282
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8678741, 34.8735657
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8260117, 27.8441162
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7666283, 18.7749863
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8255005, 22.8372803
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7252274, 28.7256699
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4502792, 25.4491119
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0447235, 17.0413284
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3481750, 17.3381042
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0659103, 19.0634995
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7987976, 18.7927170
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5019150, 19.4973946
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6060028, 20.6020126
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3052368, 20.2967453
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4390030, 27.4331055
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1334229, 22.1224518
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4881897, 17.4865265
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1275558, 24.1169357
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6722336, 18.6699448
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0986633, 23.0909882
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1789703, 32.1614456
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6504822, 24.6341934
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4974747, 24.4907074
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9266663, 28.9213638
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8315887, 33.8211060
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6021576, 37.5940247
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3055878, 39.3031311
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7748489, 24.7645264
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5752563, 24.5683670
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8033867, 15.7902489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1446

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1486231, upper bound: 9.1973273
time: 33.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1477298, upper bound: 9.1982281
time: 59.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8741837, 21.8828354
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1370430, 14.1471596
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3374710, 15.3491325
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6643143, 16.6725540
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3442459, 18.3562164
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4719162, 18.4858017
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0053101, 20.9998055
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3642731, 16.3728180
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6047363, 20.6155205
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5666809, 21.5700760
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4082642, 27.4086151
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0504990, 18.0425377
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5305099, 26.5252380
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8692780, 34.8773499
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8260193, 27.8414230
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7400208, 18.7553253
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8210068, 22.8273468
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7193222, 28.7222672
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4471436, 25.4481506
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0459290, 17.0353279
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3476067, 17.3324127
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0653610, 19.0573044
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8071899, 18.8009491
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5013123, 19.4892731
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6074448, 20.5999603
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3061371, 20.2930374
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4410782, 27.4355621
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1365738, 22.1198807
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4982529, 17.4960709
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1284180, 24.1147995
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6692886, 18.6611938
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0960312, 23.0881119
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1762390, 32.1585007
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6267090, 24.6156845
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5034027, 24.4991989
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9397583, 28.9392319
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8262939, 33.8182068
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6101837, 37.6054688
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3042450, 39.2989044
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7738876, 24.7667618
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5730591, 24.5673981
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8020897, 15.7860985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1339705, upper bound: 9.1993860
time: 26.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1318821, upper bound: 9.2014743
time: 21.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8833961, 21.8887100
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1501541, 14.1550102
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3458557, 15.3548164
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6750908, 16.6805420
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3568001, 18.3631630
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4852448, 18.4936867
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0105667, 21.0053253
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3717918, 16.3767319
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6163101, 20.6228104
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5666046, 21.5702209
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4071960, 27.4100723
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0585709, 18.0548973
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5336838, 26.5296555
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8757782, 34.8822479
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8340836, 27.8471069
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7651443, 18.7723236
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8209763, 22.8277740
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7207489, 28.7233887
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4462967, 25.4470139
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0562210, 17.0526123
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3600616, 17.3505783
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0751495, 19.0704575
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8082428, 18.8033943
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5130463, 19.5084114
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6124420, 20.6085739
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3162842, 20.3086014
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4473877, 27.4429474
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1479416, 22.1385040
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4996414, 17.4982452
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1358948, 24.1241989
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6809235, 18.6783257
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1018372, 23.0972443
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1805420, 32.1676712
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6305161, 24.6211395
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5031815, 24.4998169
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9403381, 28.9396896
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8282166, 33.8210068
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6147614, 37.6110382
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3116150, 39.3105774
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7776184, 24.7708588
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5773163, 24.5738754
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7915878, 15.7810783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 789

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1499249, upper bound: 9.2030374
time: 40.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1498771, upper bound: 9.2030876
time: 28.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8833580, 21.8887482
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1501923, 14.1549759
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3458939, 15.3547783
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6750526, 16.6805725
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3570137, 18.3629341
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4852219, 18.4937096
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0102310, 21.0056610
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3718910, 16.3766365
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6163254, 20.6227951
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5666122, 21.5702209
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4072571, 27.4100037
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0586014, 18.0548668
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5336685, 26.5296631
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8757935, 34.8822479
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8343124, 27.8468704
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7652588, 18.7722168
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8210907, 22.8276672
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7207794, 28.7233429
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4464874, 25.4468307
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0562363, 17.0525970
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3600006, 17.3506393
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0751343, 19.0704727
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8082352, 18.8034019
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5129852, 19.5084724
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6124802, 20.6085434
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3161621, 20.3087234
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4473267, 27.4430237
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1479111, 22.1385345
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4995422, 17.4983406
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1358490, 24.1242523
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6809845, 18.6782722
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1017609, 23.0973206
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1802368, 32.1679611
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6304550, 24.6212082
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5031662, 24.4998245
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9402924, 28.9397354
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8281250, 33.8210983
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6146545, 37.6111450
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3115540, 39.3106537
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7775116, 24.7709579
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5772095, 24.5739899
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7911377, 15.7815323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1498295, upper bound: 9.1974393
time: 27.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1449449, upper bound: 9.2026294
time: 30.77 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 60.03 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1432725, upper bound: 9.2042301
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1486982, upper bound: 9.1979307
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1546136, upper bound: 9.2033138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1546135, upper bound: 9.2033138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1486231, upper bound: 9.1973273
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1477298, upper bound: 9.1982281
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1339705, upper bound: 9.1993860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1318821, upper bound: 9.2014743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1499249, upper bound: 9.2030374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1498771, upper bound: 9.2030876
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1498295, upper bound: 9.1974393
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 60.03
Output dim: 2, lower bound: -9.1449449, upper bound: 9.2026294

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8710594, 21.8787231
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1426468, 14.1506271
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3513412, 15.3605652
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6718330, 16.6782265
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3607979, 18.3676491
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4791412, 18.4899063
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0179291, 21.0161705
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3669243, 16.3746605
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6162148, 20.6249466
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5611343, 21.5660553
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3809052, 27.3836441
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0607185, 18.0554123
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5331726, 26.5288239
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8759308, 34.8830261
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8344727, 27.8476334
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7649765, 18.7741013
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8226852, 22.8329391
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7233658, 28.7241898
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4464264, 25.4457779
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0553513, 17.0486603
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3593826, 17.3474197
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0822601, 19.0740089
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8069611, 18.7990379
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5087814, 19.5004501
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6205750, 20.6139832
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3145599, 20.3025780
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4458389, 27.4382019
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1449432, 22.1316605
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4978485, 17.4950523
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1337433, 24.1217499
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6843033, 18.6790657
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1003265, 23.0961304
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1789856, 32.1637878
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6462936, 24.6350250
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5088806, 24.5033035
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9386902, 28.9393463
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8362732, 33.8286438
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6083527, 37.6050415
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3101501, 39.3066864
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7917404, 24.7866440
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5880508, 24.5858154
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8091393, 15.8002167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1398

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1431207, upper bound: 9.2041981
time: 29.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1432696, upper bound: 9.2042268
time: 23.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8792343, 21.8857193
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1474609, 14.1541786
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3542709, 15.3625412
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6762657, 16.6838226
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3590622, 18.3662910
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4808693, 18.4921570
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0055161, 21.0028152
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3675842, 16.3769455
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6118011, 20.6208115
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5564728, 21.5655746
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3724213, 27.3853378
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0537262, 18.0506325
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5256653, 26.5233154
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8635712, 34.8723297
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8225403, 27.8390656
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7618408, 18.7687683
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8206177, 22.8387375
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7249146, 28.7246780
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4446869, 25.4415970
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0451431, 17.0384941
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3491516, 17.3367615
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0665054, 19.0612373
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7972336, 18.7871704
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5029755, 19.4951630
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6057281, 20.5966339
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3060837, 20.2948380
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4367523, 27.4266739
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1352234, 22.1205368
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4889832, 17.4840279
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1281204, 24.1157837
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6728821, 18.6678581
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0925522, 23.0897369
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1735687, 32.1573639
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6451721, 24.6359177
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4913101, 24.4876175
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9233398, 28.9217148
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8251953, 33.8193054
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6006317, 37.5952301
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2980194, 39.2993774
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7705460, 24.7662964
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5680847, 24.5661469
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7968674, 15.7880058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1518432, upper bound: 9.1947150
time: 25.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1459749, upper bound: 9.2006184
time: 24.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8786850, 21.8862610
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1467514, 14.1548882
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3540726, 15.3627319
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6768303, 16.6832581
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3580551, 18.3672943
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4809914, 18.4920349
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0071030, 21.0012283
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3677750, 16.3767624
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6116867, 20.6209259
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5567245, 21.5653152
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3739929, 27.3837662
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0548553, 18.0495033
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5261154, 26.5228653
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8642120, 34.8717041
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8213043, 27.8402863
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7603836, 18.7702255
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8222885, 22.8370667
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7247009, 28.7248917
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4428864, 25.4433975
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0452576, 17.0383835
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3491364, 17.3367767
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0666656, 19.0610771
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7960815, 18.7883301
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5029449, 19.4951935
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6048279, 20.5975342
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3058548, 20.2950668
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4354706, 27.4279633
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1352005, 22.1205521
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4887466, 17.4842682
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1281281, 24.1157684
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6729279, 18.6678123
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0936508, 23.0886383
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1748352, 32.1560974
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6464081, 24.6346741
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4925308, 24.4863968
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9235229, 28.9215164
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8259583, 33.8185425
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6006165, 37.5952606
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2994232, 39.2980042
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7717667, 24.7650757
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5695190, 24.5647049
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7986145, 15.7862587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1527

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1436647, upper bound: 9.2030842
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1543709, upper bound: 9.1921574
time: 28.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8788795, 21.8846588
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1544724, 14.1592712
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3507004, 15.3590012
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6612434, 16.6641846
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3880234, 18.3891525
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4685364, 18.4735832
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0153122, 21.0120316
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3604279, 16.3622131
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6213150, 20.6256638
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5842133, 21.5848923
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3970184, 27.3981934
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0668564, 18.0651207
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5345993, 26.5345993
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8775177, 34.8834991
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8441010, 27.8592072
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7754669, 18.7793808
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8234558, 22.8301163
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7471161, 28.7548828
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4450226, 25.4465637
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0480652, 17.0459595
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3631134, 17.3541718
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0758514, 19.0713768
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8064041, 18.8011665
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4990463, 19.4982605
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6097870, 20.6070938
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3271561, 20.3217049
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4461060, 27.4426956
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1384583, 22.1327744
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5062485, 17.5058746
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1370926, 24.1254654
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6819763, 18.6798210
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1018600, 23.0972672
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1676331, 32.1569672
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6183929, 24.6105652
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4887085, 24.4878159
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9316406, 28.9321136
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7991486, 33.7958984
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5873566, 37.5879669
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3140259, 39.3126221
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7698059, 24.7642517
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5651398, 24.5638885
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7752571, 15.7691917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 993

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1487

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1457288, upper bound: 9.2025036
time: 25.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1493927, upper bound: 9.1988446
time: 56.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8793449, 21.8841858
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1544189, 14.1593246
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3500443, 15.3596573
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6587334, 16.6666908
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3827744, 18.3943977
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4651489, 18.4769669
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0172653, 21.0100708
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3572693, 16.3653679
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6191635, 20.6278038
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5812836, 21.5878220
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3952942, 27.3999176
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0687943, 18.0631752
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5386276, 26.5305710
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8770142, 34.8839874
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8461609, 27.8571472
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7722015, 18.7826500
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8233109, 22.8302612
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7522278, 28.7497711
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4458466, 25.4457397
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0495682, 17.0444527
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3636475, 17.3536453
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0760727, 19.0711632
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8060074, 18.8015633
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5028915, 19.4944153
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6109543, 20.6059189
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3293762, 20.3194847
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4471283, 27.4416733
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1422424, 22.1289978
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5072784, 17.5048485
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1371689, 24.1253815
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6824265, 18.6793709
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1018600, 23.0972672
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1698608, 32.1547546
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6199341, 24.6090088
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4911804, 24.4853439
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9327545, 28.9309998
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8031006, 33.7919464
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5916748, 37.5836487
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3137207, 39.3129578
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7710114, 24.7630310
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5673370, 24.5616913
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7797050, 15.7647438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1476

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1420946, upper bound: 9.2000060
time: 32.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1467668, upper bound: 9.1953372
time: 26.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8832970, 21.8887482
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1501923, 14.1546059
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3454514, 15.3547783
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6750526, 16.6805344
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3565102, 18.3629341
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4842949, 18.4937096
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0101166, 21.0056610
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3710365, 16.3766365
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6155968, 20.6227951
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5657349, 21.5702209
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4068756, 27.4100037
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0584793, 18.0548668
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5336685, 26.5274734
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8757935, 34.8820343
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8343124, 27.8465958
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7652588, 18.7721558
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8208847, 22.8276672
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7207794, 28.7222290
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4464874, 25.4449768
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0545654, 17.0525970
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3594208, 17.3506393
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0742111, 19.0704727
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8079529, 18.8034019
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5126419, 19.5084724
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6121140, 20.6085434
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3158112, 20.3087234
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4473267, 27.4425354
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1479111, 22.1385422
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4990387, 17.4983406
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1355820, 24.1242523
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6789093, 18.6782722
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1017609, 23.0973053
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1802368, 32.1679535
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6304550, 24.6197128
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5031662, 24.4989777
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9402924, 28.9391556
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8281250, 33.8201675
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6146545, 37.6090393
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3103943, 39.3106537
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7775116, 24.7708817
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5772095, 24.5739288
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7909775, 15.7815323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1638

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1387724, upper bound: 9.1951992
time: 29.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1375625, upper bound: 9.1963789
time: 24.95 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 56.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1431207, upper bound: 9.2041981
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1432696, upper bound: 9.2042268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1518432, upper bound: 9.1947150
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1459749, upper bound: 9.2006184
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1436647, upper bound: 9.2030842
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1543709, upper bound: 9.1921574
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1457288, upper bound: 9.2025036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1493927, upper bound: 9.1988446
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1420946, upper bound: 9.2000060
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1467668, upper bound: 9.1953372
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1387724, upper bound: 9.1951992
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 56.52
Output dim: 2, lower bound: -9.1375625, upper bound: 9.1963789

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8683243, 21.8762512
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1379356, 14.1464119
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3515778, 15.3610039
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6677628, 16.6746941
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3569031, 18.3634300
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4796295, 18.4905243
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0170593, 21.0153961
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3662033, 16.3739014
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6160698, 20.6247520
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5594940, 21.5635681
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3701706, 27.3710861
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0541878, 18.0474129
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5312729, 26.5266418
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8753357, 34.8824615
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8221054, 27.8336411
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7689667, 18.7789154
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8270111, 22.8363495
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7222900, 28.7228470
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4457092, 25.4448318
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0649033, 17.0564003
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3555450, 17.3428001
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0866013, 19.0763931
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8069916, 18.7990532
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5103226, 19.5007439
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6211014, 20.6143951
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3081055, 20.2955437
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4470215, 27.4388199
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1464310, 22.1329117
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4974136, 17.4943428
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1316147, 24.1192780
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6813049, 18.6744003
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0998993, 23.0957031
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1750793, 32.1603699
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6356049, 24.6259460
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5052338, 24.5002670
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9360657, 28.9371338
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8390503, 33.8316956
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6037903, 37.6009216
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3169861, 39.3124084
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7896042, 24.7847443
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5871582, 24.5850296
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8097687, 15.8009682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1489

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1373034, upper bound: 9.1984326
time: 23.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1376588, upper bound: 9.1981542
time: 28.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8685837, 21.8759918
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1384392, 14.1459122
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3517838, 15.3607979
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6682968, 16.6741562
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3565979, 18.3637352
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4797592, 18.4904022
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0171509, 21.0152969
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3661575, 16.3739395
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6160088, 20.6248055
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5586472, 21.5644150
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3683548, 27.3729019
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0527153, 18.0488892
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5309906, 26.5269165
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8753662, 34.8824310
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8204727, 27.8352661
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7697144, 18.7780914
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8260880, 22.8372650
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7220306, 28.7231064
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4455032, 25.4450607
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0630875, 17.0582161
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3547668, 17.3435860
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0846481, 19.0783463
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8069763, 18.7990685
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5090790, 19.5019913
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6209869, 20.6145096
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3075256, 20.2961235
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4464569, 27.4393921
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1461868, 22.1331635
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4971390, 17.4946136
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1312714, 24.1196213
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6796417, 18.6760712
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0998993, 23.0956955
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1755676, 32.1598663
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6372070, 24.6243439
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5058441, 24.4996567
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9364777, 28.9367218
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8393250, 33.8314209
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6042328, 37.6004791
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3158875, 39.3134918
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7898483, 24.7844925
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5872650, 24.5849228
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8098869, 15.8008461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1511

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1421169, upper bound: 9.1946986
time: 27.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1337285, upper bound: 9.2030770
time: 29.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8641205, 21.8742523
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1373138, 14.1474838
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3489532, 15.3588905
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6727943, 16.6797981
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3527374, 18.3644028
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4738159, 18.4859085
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -20.9863129, 20.9754181
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3602753, 16.3704872
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6043930, 20.6148033
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5298767, 21.5423203
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3540268, 27.3669739
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0553284, 18.0499268
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5179291, 26.5133209
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8657684, 34.8735809
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7938538, 27.8179016
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7487564, 18.7609787
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8078690, 22.8248749
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7264481, 28.7238998
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4415588, 25.4419861
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0447845, 17.0378075
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3456039, 17.3322449
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0619354, 19.0572472
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7947693, 18.7867432
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5027237, 19.4947739
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6037521, 20.5962753
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3011169, 20.2892303
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4366531, 27.4289932
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1326904, 22.1164093
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4883881, 17.4830322
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1333923, 24.1192856
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6732407, 18.6680527
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0850906, 23.0780334
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1478882, 32.1234512
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6295547, 24.6131973
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4786224, 24.4692459
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9146729, 28.9108047
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8067474, 33.7950134
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5931091, 37.5862732
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2923737, 39.2891693
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7595825, 24.7479019
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5579071, 24.5502167
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7895164, 15.7729683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1366

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 940

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1375527, upper bound: 9.2030604
time: 23.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1436410, upper bound: 9.1969594
time: 27.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8787270, 21.8845596
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1540222, 14.1589699
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3502579, 15.3586960
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6604958, 16.6636620
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3874359, 18.3887596
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4677429, 18.4730225
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0152130, 21.0119476
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3596344, 16.3616409
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6209335, 20.6254158
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5838394, 21.5846329
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3967209, 27.3979111
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0668335, 18.0651016
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5345612, 26.5343323
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8772125, 34.8832855
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8440399, 27.8588333
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7754593, 18.7793732
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8225555, 22.8295288
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7471542, 28.7545013
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4447098, 25.4460068
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0477600, 17.0454788
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3626785, 17.3535194
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0755463, 19.0709419
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8059158, 18.8003197
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4985123, 19.4974937
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6094360, 20.6065292
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3265762, 20.3207817
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4453278, 27.4415359
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1378708, 22.1319351
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5058823, 17.5053253
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1368637, 24.1251221
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6817398, 18.6794205
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1017761, 23.0971527
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1674500, 32.1566467
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6183014, 24.6104660
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4886703, 24.4877472
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9316101, 28.9320679
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7988129, 33.7953644
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5870514, 37.5875092
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3139954, 39.3125458
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7696838, 24.7641296
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5650177, 24.5636749
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7751389, 15.7689781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1447

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1398

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1455777, upper bound: 9.2024724
time: 27.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1457258, upper bound: 9.2025004
time: 25.96 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 54.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1373034, upper bound: 9.1984326
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1376588, upper bound: 9.1981542
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1421169, upper bound: 9.1946986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1337285, upper bound: 9.2030770
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1375527, upper bound: 9.2030604
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1436410, upper bound: 9.1969594
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1455777, upper bound: 9.2024724
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 54.99
Output dim: 2, lower bound: -9.1457258, upper bound: 9.2025004

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8679543, 21.8757324
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1381683, 14.1457176
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3509979, 15.3602486
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6674232, 16.6735191
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3564377, 18.3636513
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4785118, 18.4894943
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0164719, 21.0152130
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3651314, 16.3732300
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6150246, 20.6241112
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5568466, 21.5631027
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3655167, 27.3702698
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0523148, 18.0484543
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5305481, 26.5262604
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8751221, 34.8822784
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8197556, 27.8342819
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7694855, 18.7779274
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8247910, 22.8370056
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7211914, 28.7220230
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4436951, 25.4424820
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0629578, 17.0580788
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3545761, 17.3430710
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0844574, 19.0781174
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8060913, 18.7975464
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5089340, 19.5017624
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6205215, 20.6136780
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3065338, 20.2947693
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4444275, 27.4365387
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1453781, 22.1320496
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4971085, 17.4945831
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1312103, 24.1195221
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6793518, 18.6759033
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0992813, 23.0950623
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1751099, 32.1592484
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6366577, 24.6235504
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.5048065, 24.4981613
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9357147, 28.9356308
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8380585, 33.8296204
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.6028748, 37.5985718
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3153534, 39.3129730
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7893143, 24.7838593
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5865173, 24.5841141
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8089600, 15.7999077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1617

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1301480, upper bound: 9.2026094
time: 24.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1332568, upper bound: 9.1995038
time: 22.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8664246, 21.8793869
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1302986, 14.1423187
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3436317, 15.3543549
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6724892, 16.6791077
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3525772, 18.3646812
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4750061, 18.4867592
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -20.9616089, 20.9463272
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3620567, 16.3730354
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6045876, 20.6150017
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5228882, 21.5392914
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3349533, 27.3509598
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0524826, 18.0489044
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5039825, 26.4964828
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8629456, 34.8703613
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7821732, 27.8082047
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7478409, 18.7600250
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7997208, 22.8192902
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7093506, 28.7036591
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4427414, 25.4431839
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0407333, 17.0356064
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3508759, 17.3372078
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0588608, 19.0563011
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7943878, 18.7861443
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4945374, 19.4885406
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5967560, 20.5905914
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3018494, 20.2899628
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4391479, 27.4309158
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1304779, 22.1146393
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4886322, 17.4834328
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1333160, 24.1192245
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6658554, 18.6617508
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0717468, 23.0620728
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1329041, 32.1056213
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6076889, 24.5869446
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4664612, 24.4548569
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8970642, 28.8896637
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7880249, 33.7730560
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5738525, 37.5633545
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2837524, 39.2791290
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7362671, 24.7203217
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5476913, 24.5381699
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7831917, 15.7638664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1493

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1518

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1335068, upper bound: 9.2026865
time: 28.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1371791, upper bound: 9.1990142
time: 24.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8760185, 21.8821106
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1493073, 14.1547585
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3504829, 15.3591385
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6564255, 16.6601257
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3835258, 18.3845596
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4682236, 18.4736290
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0143433, 21.0111694
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3588982, 16.3608665
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6208000, 20.6252174
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5821762, 21.5821228
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3859863, 27.3853455
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0603104, 18.0570908
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5326385, 26.5321274
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8766174, 34.8827515
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8316803, 27.8448334
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7794495, 18.7841911
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8268814, 22.8329544
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7460861, 28.7531662
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4439926, 25.4450684
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0572968, 17.0532227
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3588715, 17.3489189
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0798874, 19.0733223
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8059464, 18.8003349
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5000763, 19.4978180
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6099701, 20.6069641
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3201065, 20.3137550
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4465332, 27.4421692
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1393509, 22.1331940
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5054398, 17.5046234
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1347351, 24.1226425
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6787415, 18.6747589
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1013260, 23.0967178
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1636047, 32.1532669
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6076660, 24.6013947
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4850159, 24.4847031
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9290161, 28.9298935
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8015747, 33.7984009
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5824890, 37.5834045
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3208008, 39.3182983
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7675171, 24.7622223
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5641327, 24.5628967
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7757683, 15.7697372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1259605, upper bound: 9.1945260
time: 20.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1376311, upper bound: 9.1828641
time: 20.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8762703, 21.8818588
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1498108, 14.1542549
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3506889, 15.3589325
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6569595, 16.6595917
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3832283, 18.3848610
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4683456, 18.4735069
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0144501, 21.0110703
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3588524, 16.3609085
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6207390, 20.6252747
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5813217, 21.5829697
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3841705, 27.3871689
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0588303, 18.0585632
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5323563, 26.5324097
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8766632, 34.8827209
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8300476, 27.8464661
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7802048, 18.7833633
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8259583, 22.8338776
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7458267, 28.7534180
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4437790, 25.4453049
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0554810, 17.0550385
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3580856, 17.3497047
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0779343, 19.0752678
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8059311, 18.8003502
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4988327, 19.4990654
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6098557, 20.6070786
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3195267, 20.3143349
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4459686, 27.4427414
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1391068, 22.1334457
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5051727, 17.5048943
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1343994, 24.1229858
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6770706, 18.6764297
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1013336, 23.0967178
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1640930, 32.1527634
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6092529, 24.5998001
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4856110, 24.4841003
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9294281, 28.9294815
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8018341, 33.7981415
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5829315, 37.5829620
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3197327, 39.3193665
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7677612, 24.7619705
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5642395, 24.5627899
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7758904, 15.7696152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 810

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1261099, upper bound: 9.1945539
time: 28.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1377803, upper bound: 9.1828919
time: 24.45 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 54.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1301480, upper bound: 9.2026094
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1332568, upper bound: 9.1995038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1335068, upper bound: 9.2026865
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1371791, upper bound: 9.1990142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1259605, upper bound: 9.1945260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1376311, upper bound: 9.1828641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1261099, upper bound: 9.1945539
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 54.77
Output dim: 2, lower bound: -9.1377803, upper bound: 9.1828919

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8508682, 21.8622360
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0987816, 14.1132393
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3554993, 15.3635406
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6665306, 16.6709480
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3330727, 18.3457108
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4767952, 18.4862671
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -20.9793243, 20.9713402
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3275833, 16.3424606
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5879517, 20.6013794
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5429459, 21.5532608
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3402863, 27.3490829
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0474319, 18.0473595
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5176849, 26.5108643
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8458710, 34.8464050
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7812424, 27.8035278
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7764854, 18.7823563
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7819214, 22.8020935
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7220001, 28.7217102
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4297104, 25.4331589
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0665436, 17.0636711
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3531761, 17.3409119
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0969620, 19.0974922
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7974548, 18.7877998
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5099182, 19.5027542
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6113663, 20.6075745
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2946091, 20.2853165
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4354553, 27.4224701
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1460419, 22.1326141
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4789352, 17.4730835
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1169357, 24.1102219
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6757584, 18.6777420
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0749817, 23.0661697
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1658783, 32.1449127
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6138535, 24.5959167
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4723434, 24.4595108
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8807068, 28.8702850
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8278656, 33.8151016
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5652008, 37.5537567
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3058014, 39.3012848
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7942963, 24.7858963
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5511169, 24.5423889
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8173676, 15.8016548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1513

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1290608, upper bound: 9.1883054
time: 24.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1190179, upper bound: 9.2015319
time: 22.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8663712, 21.8793488
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1290016, 14.1412811
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3431129, 15.3539505
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6712494, 16.6782417
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3522491, 18.3644257
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4743500, 18.4862785
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -20.9588776, 20.9436836
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3598709, 16.3713150
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6033211, 20.6139908
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5226059, 21.5390701
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3338547, 27.3502502
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0511551, 18.0473404
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5038910, 26.4963379
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8626709, 34.8699493
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7819824, 27.8080521
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7459030, 18.7580681
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7971573, 22.8172913
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7090225, 28.7031097
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4420776, 25.4424057
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0399094, 17.0344276
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3506241, 17.3369026
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0581970, 19.0554466
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7920990, 18.7832375
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4937210, 19.4874115
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5961380, 20.5897980
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3003769, 20.2880859
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4373779, 27.4286575
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1297989, 22.1136703
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4873276, 17.4815903
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1328278, 24.1185684
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6653442, 18.6611557
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.0710678, 23.0613480
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1318970, 32.1043243
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6073837, 24.5865173
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4654083, 24.4534836
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8964539, 28.8888855
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7865601, 33.7711792
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5736542, 37.5630951
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2833710, 39.2786255
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7352448, 24.7194214
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5472031, 24.5377045
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.7820549, 15.7623672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1527

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1324754, upper bound: 9.1964124
time: 30.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1272375, upper bound: 9.2016513
time: 23.11 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 55.43 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 14, time: 55.43
Output dim: 2, lower bound: -9.1290608, upper bound: 9.1883054
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 14, time: 55.43
Output dim: 2, lower bound: -9.1190179, upper bound: 9.2015319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 14, time: 55.43
Output dim: 2, lower bound: -9.1324754, upper bound: 9.1964124
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 14, time: 55.43
Output dim: 2, lower bound: -9.1272375, upper bound: 9.2016513

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 43.72 + 2609.86 = 2653.58 seconds
