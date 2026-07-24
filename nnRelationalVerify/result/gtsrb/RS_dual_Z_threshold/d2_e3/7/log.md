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
execution time: IAR + RelationalAnalysis = 2.38 + 40.84 = 43.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -9.2114816, upper bound: 9.2114816

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2005763, upper bound: 9.2111152
time: 29.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2111152, upper bound: 9.2005763
time: 22.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 51.98 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 51.98
Output dim: 2, lower bound: -9.2005763, upper bound: 9.2111152
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 51.98
Output dim: 2, lower bound: -9.2111152, upper bound: 9.2005763

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8932190, 21.8945923
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1494904, 14.1532288
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3874779, 15.3872566
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6970711, 16.6971741
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3810120, 18.3860970
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4975700, 18.4990196
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0149536, 21.0143547
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3647690, 16.3689575
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6096039, 20.6136284
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5584183, 21.5626068
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4042740, 27.4063644
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0883865, 18.0883865
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5516205, 26.5514374
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8948517, 34.8925629
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8242035, 27.8315735
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7786102, 18.7811813
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8433609, 22.8452148
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7552109, 28.7553329
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4718628, 25.4744415
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0717926, 17.0716515
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3928108, 17.3928757
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1206894, 19.1244278
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8380203, 18.8384933
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5246887, 19.5222473
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6392822, 20.6393585
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3424988, 20.3428345
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4695282, 27.4695206
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1549911, 22.1519318
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5110245, 17.5107689
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1729202, 24.1743011
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7109680, 18.7124405
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1117935, 23.1089706
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1758728, 32.1707993
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6317520, 24.6259766
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4925842, 24.4868011
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9295044, 28.9240570
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8379669, 33.8361511
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5967255, 37.5916748
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3107910, 39.3099213
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7861023, 24.7859802
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5637360, 24.5581207
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8588715, 15.8539124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1821830, upper bound: 9.2103380
time: 25.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1999874, upper bound: 9.1927694
time: 29.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8945923, 21.8932190
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1532288, 14.1494904
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3872566, 15.3874741
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6971779, 16.6970673
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3860931, 18.3810120
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4990196, 18.4975700
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0143585, 21.0149498
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3689575, 16.3647690
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.6136246, 20.6096039
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5625992, 21.5584183
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.4063644, 27.4042664
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0883865, 18.0883865
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5514374, 26.5516205
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8925629, 34.8948517
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.8315735, 27.8242035
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7811813, 18.7786102
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8452072, 22.8433685
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7553329, 28.7552109
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4744415, 25.4718628
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0716476, 17.0717964
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3928719, 17.3928070
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1244278, 19.1206894
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8384933, 18.8380203
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.5222473, 19.5246887
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6393585, 20.6392822
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3428345, 20.3424950
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4695282, 27.4695358
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1519318, 22.1549911
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5107651, 17.5110207
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1743011, 24.1729202
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7124405, 18.7109718
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1089706, 23.1117935
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1707916, 32.1758652
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6259842, 24.6317596
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4868011, 24.4925919
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9240417, 28.9295044
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8361664, 33.8379669
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5916748, 37.5967255
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3099060, 39.3108063
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7859802, 24.7861023
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5581207, 24.5637360
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8539124, 15.8588676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1927694, upper bound: 9.1999874
time: 25.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2103380, upper bound: 9.1821830
time: 29.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 56.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 56.70
Output dim: 2, lower bound: -9.1821830, upper bound: 9.2103380
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 56.70
Output dim: 2, lower bound: -9.1999874, upper bound: 9.1927694
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 56.70
Output dim: 2, lower bound: -9.1927694, upper bound: 9.1999874
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 56.70
Output dim: 2, lower bound: -9.2103380, upper bound: 9.1821830

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8836060, 21.8862534
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1250343, 14.1321030
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3725700, 15.3748627
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6731606, 16.6770325
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3560181, 18.3664055
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4743805, 18.4795113
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0083466, 21.0077057
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3378105, 16.3457031
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5795212, 20.5881195
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5390701, 21.5450897
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3797455, 27.3846359
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0828934, 18.0825119
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5529404, 26.5520248
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8941498, 34.8920593
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7763977, 27.7907715
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7516365, 18.7585526
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8231812, 22.8271790
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7579041, 28.7582703
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4704895, 25.4731140
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0497742, 17.0453453
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3728638, 17.3691063
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1105804, 19.1132584
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8231659, 18.8217163
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4965668, 19.4890213
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6259918, 20.6236496
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3154526, 20.3105698
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4509888, 27.4481964
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1226807, 22.1138229
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.5009766, 17.4987488
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1670074, 24.1677933
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6966782, 18.6960983
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1113586, 23.1084518
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1471100, 32.1371307
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6259613, 24.6197357
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4754181, 24.4675522
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9160309, 28.9084625
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8079834, 33.8006592
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5776215, 37.5694275
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.3007355, 39.2982941
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7847290, 24.7845840
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5509033, 24.5433807
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8765411, 15.8672295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1783168, upper bound: 9.2103270
time: 28.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1821575, upper bound: 9.2026933
time: 31.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8862610, 21.8836060
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1320992, 14.1250305
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3748589, 15.3725739
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6770363, 16.6731644
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3664093, 18.3560219
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4795074, 18.4743843
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0077057, 21.0083466
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3456993, 16.3378067
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5881195, 20.5795212
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5450897, 21.5390778
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3846436, 27.3797455
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0825119, 18.0828896
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5520248, 26.5529404
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8920593, 34.8941498
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7907715, 27.7763901
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7585564, 18.7516365
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.8271790, 22.8231812
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7582703, 28.7579041
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4731140, 25.4704895
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0453415, 17.0497742
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3691101, 17.3728676
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1132507, 19.1105804
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8217163, 18.8231659
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4890213, 19.4965668
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6236572, 20.6259918
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3105698, 20.3154564
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4481964, 27.4509888
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1138229, 22.1226807
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4987488, 17.5009766
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1677933, 24.1670074
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6960983, 18.6966782
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1084518, 23.1113586
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1371307, 32.1471100
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6197357, 24.6259689
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4675446, 24.4754181
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.9084778, 28.9160309
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.8006592, 33.8079834
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5694275, 37.5776215
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2982941, 39.3007507
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7845917, 24.7847290
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5433807, 24.5509109
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8672295, 15.8765411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2026933, upper bound: 9.1821575
time: 31.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2103270, upper bound: 9.1783168
time: 28.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 62.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 62.42
Output dim: 2, lower bound: -9.1783168, upper bound: 9.2103270
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 62.42
Output dim: 2, lower bound: -9.1821575, upper bound: 9.2026933
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 62.42
Output dim: 2, lower bound: -9.2026933, upper bound: 9.1821575
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 62.42
Output dim: 2, lower bound: -9.2103270, upper bound: 9.1783168

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8801079, 21.8832779
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1138153, 14.1226768
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3743057, 15.3766556
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6736679, 16.6775475
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3547096, 18.3661118
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4711533, 18.4767761
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0174789, 21.0175476
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3171654, 16.3284760
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5618668, 20.5731850
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5163116, 21.5257416
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3419495, 27.3529968
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0939941, 18.0961189
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5519791, 26.5510788
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8904419, 34.8863983
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7329483, 27.7541046
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7583466, 18.7653122
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7861023, 22.7958755
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7646255, 28.7648544
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4703903, 25.4730148
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0507355, 17.0462646
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3762550, 17.3723259
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1240845, 19.1279449
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8112640, 18.8076248
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4936066, 19.4854088
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6249466, 20.6224518
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3114014, 20.3059158
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4393387, 27.4343033
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1141205, 22.1035233
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4948273, 17.4912758
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1781998, 24.1805954
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7083969, 18.7084541
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1166840, 23.1143494
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1278534, 32.1140289
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6157532, 24.6076813
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4495926, 24.4365692
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8960571, 28.8844986
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7909546, 33.7802277
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5620880, 37.5507812
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2995453, 39.2969513
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7875824, 24.7889328
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5455627, 24.5373840
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8753624, 15.8658905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1764121, upper bound: 9.1819658
time: 27.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1499534, upper bound: 9.2084229
time: 39.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8806267, 21.8827515
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1154861, 14.1208878
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3743668, 15.3765793
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6736755, 16.6775284
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3557320, 18.3650894
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4716492, 18.4762764
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0180206, 21.0168304
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3204918, 16.3250618
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5645981, 20.5704536
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5197220, 21.5223236
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3481140, 27.3468399
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0964355, 18.0936203
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5519714, 26.5510635
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8884888, 34.8883057
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7397232, 27.7473297
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7583923, 18.7652664
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7914581, 22.7901001
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7644882, 28.7649460
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4703903, 25.4730072
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0506821, 17.0463028
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3760033, 17.3724861
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1251221, 19.1267548
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8090744, 18.8098221
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4929581, 19.4860611
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6247864, 20.6225815
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3107986, 20.3065033
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4370956, 27.4365387
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1123810, 22.1052628
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4934998, 17.4926033
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1797180, 24.1789932
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7085724, 18.7078133
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1170959, 23.1137772
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1240082, 32.1178741
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6139069, 24.6094437
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4444351, 24.4416351
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8920593, 28.8884811
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7875671, 33.7836304
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5589752, 37.5538940
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2994232, 39.2970123
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7889252, 24.7874374
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5449219, 24.5380325
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8752022, 15.8660507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1802545, upper bound: 9.1743293
time: 26.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1537921, upper bound: 9.2007870
time: 25.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8827629, 21.8806305
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1208878, 14.1154900
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3765793, 15.3743668
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6775284, 16.6736755
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3650932, 18.3557281
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4762802, 18.4716492
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0168381, 21.0180206
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3250618, 16.3204956
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5704575, 20.5645943
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5223236, 21.5197220
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3468475, 27.3481064
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0936127, 18.0964394
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5510635, 26.5519714
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8883057, 34.8884888
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7473221, 27.7397232
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7652740, 18.7583961
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7901001, 22.7914581
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7649460, 28.7644882
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4730072, 25.4703903
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0463028, 17.0506821
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3724861, 17.3760033
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1267548, 19.1251221
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8098221, 18.8090744
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4860611, 19.4929581
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6225815, 20.6247864
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3065033, 20.3107986
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4365311, 27.4370956
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1052704, 22.1123810
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4925995, 17.4935036
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1789932, 24.1797180
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7078171, 18.7085686
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1137772, 23.1170959
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1178741, 32.1240082
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6094360, 24.6139069
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4416275, 24.4444351
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8884888, 28.8920670
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7836304, 33.7875671
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5538788, 37.5589752
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2970123, 39.2994080
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7874298, 24.7889175
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5380249, 24.5449219
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8660507, 15.8752022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2007870, upper bound: 9.1537921
time: 24.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1743293, upper bound: 9.1802545
time: 23.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8832817, 21.8801117
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1226730, 14.1138153
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3766556, 15.3743095
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6775513, 16.6736679
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3661156, 18.3547096
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4767761, 18.4711494
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0175476, 21.0174789
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3284721, 16.3171692
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5731888, 20.5618629
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5257416, 21.5163116
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3529968, 27.3419495
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0961151, 18.0939980
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5510788, 26.5519791
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8863983, 34.8904419
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7540970, 27.7329407
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7653198, 18.7583504
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7958755, 22.7861023
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7648544, 28.7646255
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4730148, 25.4703903
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0462646, 17.0507317
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3723259, 17.3762512
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1279449, 19.1240845
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8076248, 18.8112640
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4854050, 19.4936066
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6224518, 20.6249466
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3059158, 20.3114014
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4343033, 27.4393311
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1035233, 22.1141205
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4912720, 17.4948311
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1805954, 24.1781998
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7084503, 18.7083931
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1143494, 23.1166840
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1140289, 32.1278534
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6076813, 24.6157455
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4365616, 24.4495926
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8844910, 28.8960495
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7802429, 33.7909546
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5507660, 37.5620880
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2969513, 39.2995453
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7889252, 24.7875748
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5373840, 24.5455704
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8658905, 15.8753624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2084230, upper bound: 9.1499534
time: 23.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1819658, upper bound: 9.1764121
time: 25.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 51.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1764121, upper bound: 9.1819658
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1499534, upper bound: 9.2084229
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1802545, upper bound: 9.1743293
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1537921, upper bound: 9.2007870
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.2007870, upper bound: 9.1537921
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1743293, upper bound: 9.1802545
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.2084230, upper bound: 9.1499534
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 51.46
Output dim: 2, lower bound: -9.1819658, upper bound: 9.1764121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8777504, 21.8829346
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1123962, 14.1225166
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3711014, 15.3764114
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6707649, 16.6774597
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3523788, 18.3659515
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4663429, 18.4764061
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0170898, 21.0183487
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3140182, 16.3282318
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5585861, 20.5729370
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5142899, 21.5246124
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3408966, 27.3521423
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0934982, 18.0948029
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5563202, 26.5503464
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8890686, 34.8863983
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7333298, 27.7535477
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7568970, 18.7648926
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7844620, 22.7971649
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7641220, 28.7631607
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4700089, 25.4680710
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0506134, 17.0459862
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3761330, 17.3697548
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1238785, 19.1262703
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8109360, 18.8055229
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4934158, 19.4830589
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6246338, 20.6208878
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3112335, 20.3035965
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4385757, 27.4276505
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1138840, 22.1003952
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4947968, 17.4914551
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1778412, 24.1786041
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7083130, 18.7087250
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1174011, 23.1142883
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1278687, 32.1132278
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6154327, 24.6035385
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4494095, 24.4340134
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8958588, 28.8826370
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7906494, 33.7761154
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5617981, 37.5461578
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2990265, 39.2969360
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7875214, 24.7887573
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5455627, 24.5373611
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8753662, 15.8658257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1769

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1491670, upper bound: 9.1767735
time: 30.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1199525, upper bound: 9.2070565
time: 27.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8829384, 21.8777466
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1225128, 14.1123924
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3764114, 15.3711014
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6774635, 16.6707649
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3659515, 18.3523788
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4764061, 18.4663391
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0183411, 21.0170898
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3282318, 16.3140182
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5729370, 20.5585899
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5246124, 21.5142899
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3521423, 27.3408890
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0948029, 18.0934982
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5503464, 26.5563202
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8863983, 34.8890686
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7535477, 27.7333298
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7648926, 18.7568970
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7971649, 22.7844620
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7631607, 28.7641220
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4680710, 25.4700089
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0459900, 17.0506096
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3697548, 17.3761330
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1262665, 19.1238747
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8055191, 18.8109398
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4830627, 19.4934158
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6208878, 20.6246338
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3035965, 20.3112297
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4276505, 27.4385834
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1003952, 22.1138840
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4914551, 17.4947968
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1786118, 24.1778412
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7087250, 18.7083092
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1142883, 23.1174011
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1132202, 32.1278610
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6035461, 24.6154327
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4340134, 24.4494019
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8826294, 28.8958664
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7761230, 33.7906570
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5461578, 37.5617981
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2969360, 39.2990265
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7887573, 24.7875290
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5373535, 24.5455551
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8658257, 15.8753662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1769

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2070565, upper bound: 9.1199525
time: 21.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1767735, upper bound: 9.1491670
time: 24.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 48.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 48.75
Output dim: 2, lower bound: -9.1491670, upper bound: 9.1767735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 48.75
Output dim: 2, lower bound: -9.1199525, upper bound: 9.2070565
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 48.75
Output dim: 2, lower bound: -9.2070565, upper bound: 9.1199525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 48.75
Output dim: 2, lower bound: -9.1767735, upper bound: 9.1491670

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8687286, 21.8788986
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1075211, 14.1201706
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3622017, 15.3720551
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6658401, 16.6747131
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3451843, 18.3623657
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4560280, 18.4711685
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0152969, 21.0166321
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3097000, 16.3260498
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5529556, 20.5702019
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5089111, 21.5209122
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3383789, 27.3502655
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0873337, 18.0860786
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5572433, 26.5451660
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8843842, 34.8844299
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7327347, 27.7531052
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7478065, 18.7584457
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7824173, 22.7987061
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7641068, 28.7631607
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4650269, 25.4593964
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0487823, 17.0436249
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3757057, 17.3630943
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1191101, 19.1165085
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8093796, 18.8025780
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4900131, 19.4763412
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6222076, 20.6175842
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3077850, 20.2967224
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4300613, 27.4102554
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1096649, 22.0920792
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4946060, 17.4920540
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1733551, 24.1693878
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7072144, 18.7078590
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1197205, 23.1138763
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1272278, 32.1098785
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6105347, 24.5939026
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4468765, 24.4290619
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8944244, 28.8795853
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7856750, 33.7662964
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5567322, 37.5358734
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2986908, 39.2967072
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7856903, 24.7846756
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5447388, 24.5362473
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8758965, 15.8641396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1151864, upper bound: 9.2066703
time: 26.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1194860, upper bound: 9.2003068
time: 22.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8789062, 21.8687286
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1201706, 14.1075211
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3720512, 15.3622017
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6747131, 16.6658401
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3623657, 18.3451843
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4711723, 18.4560280
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0166321, 21.0152969
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3260498, 16.3097038
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5702057, 20.5529556
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5209122, 21.5089111
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3502655, 27.3383789
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0860748, 18.0873337
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5451660, 26.5572433
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8844299, 34.8843842
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7531052, 27.7327347
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7584496, 18.7478104
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7987061, 22.7824173
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7631607, 28.7641144
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4593964, 25.4650269
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0436249, 17.0487785
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3631020, 17.3756981
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1165085, 19.1191025
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8025818, 18.8093834
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4763412, 19.4900093
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6175842, 20.6222076
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2967224, 20.3077888
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4102554, 27.4300537
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0920792, 22.1096649
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4920578, 17.4946098
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1693878, 24.1733551
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7078552, 18.7072182
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1138763, 23.1197205
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1098785, 32.1272278
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5939026, 24.6105423
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4290543, 24.4468842
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8795929, 28.8944321
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7662964, 33.7856750
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5358734, 37.5567322
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2967072, 39.2986908
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7846832, 24.7856979
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5362549, 24.5447464
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8641396, 15.8758965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2003068, upper bound: 9.1194860
time: 26.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2066703, upper bound: 9.1151864
time: 35.86 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 64.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.42
Output dim: 2, lower bound: -9.1151864, upper bound: 9.2066703
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 64.42
Output dim: 2, lower bound: -9.1194860, upper bound: 9.2003068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 64.42
Output dim: 2, lower bound: -9.2003068, upper bound: 9.1194860
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.42
Output dim: 2, lower bound: -9.2066703, upper bound: 9.1151864

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8646545, 21.8763809
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1016464, 14.1174049
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3614769, 15.3714371
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6658287, 16.6743546
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3411179, 18.3600159
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4540749, 18.4702225
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0145187, 21.0158539
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3024940, 16.3227272
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5468674, 20.5673790
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5035934, 21.5184097
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3311081, 27.3472672
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0852356, 18.0854607
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5563202, 26.5442657
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8834534, 34.8818207
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7200089, 27.7475510
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7476082, 18.7581139
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7712021, 22.7933350
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7637711, 28.7628555
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4646072, 25.4593353
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0487518, 17.0436440
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3743820, 17.3610458
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1183624, 19.1175117
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8068466, 18.7984123
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4887619, 19.4740295
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6221008, 20.6174316
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3069153, 20.2954712
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4270859, 27.4044037
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1079254, 22.0879517
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4932785, 17.4897537
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1720505, 24.1689148
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7067490, 18.7092247
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1185684, 23.1116028
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1235657, 32.1014252
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6071930, 24.5868073
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4430542, 24.4202652
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8912506, 28.8724136
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7833557, 33.7619629
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5539093, 37.5295715
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2975616, 39.2954712
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7854462, 24.7846146
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5424194, 24.5326614
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8752670, 15.8619804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1022467, upper bound: 9.2057892
time: 20.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1144189, upper bound: 9.1936912
time: 32.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8763885, 21.8646545
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1174088, 14.1016426
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3714409, 15.3614807
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6743584, 16.6658249
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3600082, 18.3411140
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4702187, 18.4540787
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0158539, 21.0145187
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3227272, 16.3024902
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5673752, 20.5468674
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5184097, 21.5035934
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3472672, 27.3311005
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0854645, 18.0852318
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5442657, 26.5563202
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8818207, 34.8834534
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7475510, 27.7200089
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7581139, 18.7476120
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7933350, 22.7712097
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7628555, 28.7637711
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4593353, 25.4646072
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0436401, 17.0487480
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3610458, 17.3743858
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1175079, 19.1183586
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7984161, 18.8068504
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4740295, 19.4887619
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6174316, 20.6221008
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2954712, 20.3069153
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4044113, 27.4270859
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0879517, 22.1079254
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4897537, 17.4932785
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1689148, 24.1720505
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7092209, 18.7067451
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1116028, 23.1185684
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1014252, 32.1235504
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5868073, 24.6072083
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4202728, 24.4430542
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8724060, 28.8912430
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7619629, 33.7833557
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5295715, 37.5539093
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2954712, 39.2975616
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7846222, 24.7854385
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5326538, 24.5424271
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8619766, 15.8752708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1936912, upper bound: 9.1144189
time: 31.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2057892, upper bound: 9.1022466
time: 30.26 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 63.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 63.78
Output dim: 2, lower bound: -9.1022467, upper bound: 9.2057892
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 63.78
Output dim: 2, lower bound: -9.1144189, upper bound: 9.1936912
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 63.78
Output dim: 2, lower bound: -9.1936912, upper bound: 9.1144189
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 63.78
Output dim: 2, lower bound: -9.2057892, upper bound: 9.1022466

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8559265, 21.8722382
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0910835, 14.1122894
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3577309, 15.3696213
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6638336, 16.6734314
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3338394, 18.3559341
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4492455, 18.4678574
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0127106, 21.0135269
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2951050, 16.3190231
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5391693, 20.5634995
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5002060, 21.5162659
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3264160, 27.3442993
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0839920, 18.0840950
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5553436, 26.5437164
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8819580, 34.8806152
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7032852, 27.7384186
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7430992, 18.7555046
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7589951, 22.7875137
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7582397, 28.7603607
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4640274, 25.4601593
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0454788, 17.0365562
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3688507, 17.3494911
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1157837, 19.1124611
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.8028259, 18.7899590
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4843826, 19.4654732
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6200333, 20.6128922
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.3030396, 20.2875175
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.4218216, 27.3933258
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.1026154, 22.0775223
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4908524, 17.4853859
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1695633, 24.1643906
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7039795, 18.7051964
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1164246, 23.1071548
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.1178436, 32.0891266
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6043625, 24.5829849
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4405365, 24.4152985
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8892059, 28.8688889
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7815552, 33.7589722
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5523987, 37.5272064
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2950439, 39.2914124
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7853470, 24.7845535
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5406799, 24.5300446
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8707581, 15.8536339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.0910902, upper bound: 9.2053182
time: 24.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1012028, upper bound: 9.1895152
time: 27.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8722382, 21.8559341
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.1122932, 14.0910873
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3696175, 15.3577309
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6734314, 16.6638336
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3559341, 18.3338394
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4678612, 18.4492455
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0135269, 21.0127106
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3190231, 16.2951050
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5634995, 20.5391731
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5162659, 21.5002060
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3442993, 27.3264084
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0840988, 18.0839882
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5437164, 26.5553436
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8806305, 34.8819580
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.7384109, 27.7032852
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7555046, 18.7430954
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7875137, 22.7589951
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7603607, 28.7582397
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4601593, 25.4640274
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0365601, 17.0454788
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3494873, 17.3688507
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.1124573, 19.1157875
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7899551, 18.8028297
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4654694, 19.4843826
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6128922, 20.6200333
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2875214, 20.3030396
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3933182, 27.4218216
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0775223, 22.1026154
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4853821, 17.4908524
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1643906, 24.1695633
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.7052002, 18.7039757
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1071548, 23.1164322
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0891113, 32.1178436
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5829849, 24.6043701
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4152985, 24.4405365
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8688812, 28.8892136
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7589722, 33.7815552
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5272064, 37.5523987
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2914124, 39.2950439
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7845535, 24.7853470
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5300446, 24.5406799
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8536301, 15.8707542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1895152, upper bound: 9.1012028
time: 31.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2053182, upper bound: 9.0910902
time: 27.94 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 61.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 61.14
Output dim: 2, lower bound: -9.0910902, upper bound: 9.2053182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 61.14
Output dim: 2, lower bound: -9.1012028, upper bound: 9.1895152
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 61.14
Output dim: 2, lower bound: -9.1895152, upper bound: 9.1012028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 61.14
Output dim: 2, lower bound: -9.2053182, upper bound: 9.0910902

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8357468, 21.8553009
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0627632, 14.0884933
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3464355, 15.3600807
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6500587, 16.6620712
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3173790, 18.3421097
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4347458, 18.4557953
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0137634, 21.0147133
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2747345, 16.3017845
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5174332, 20.5450249
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.4863434, 21.5039062
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3005447, 27.3207626
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0811958, 18.0810089
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5447769, 26.5334396
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8776093, 34.8766174
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6504364, 27.6929855
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7300949, 18.7445679
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7235107, 22.7571259
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7477493, 28.7512589
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4679718, 25.4645386
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0225601, 17.0087662
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3423004, 17.3173981
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0990295, 19.0928001
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7787476, 18.7612190
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4620285, 19.4386520
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6043320, 20.5940781
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2776413, 20.2569466
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3964844, 27.3631058
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0757065, 22.0450439
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4765015, 17.4683342
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1586456, 24.1515427
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6871262, 18.6857605
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1164017, 23.1071167
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0877228, 32.0533295
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6067276, 24.5850830
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4290924, 24.4021072
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8813782, 28.8602219
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7658081, 33.7409973
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5491486, 37.5236969
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2838440, 39.2784729
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7881622, 24.7881851
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5389252, 24.5281296
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8647614, 15.8454247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.0881116, upper bound: 9.2044973
time: 25.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.0903758, upper bound: 9.2023611
time: 25.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8553009, 21.8357468
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0884972, 14.0627632
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3600769, 15.3464355
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6620674, 16.6500549
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3421059, 18.3173790
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4557953, 18.4347420
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0147095, 21.0137596
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3017883, 16.2747345
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5450211, 20.5174408
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5039062, 21.4863434
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3207626, 27.3005371
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0810051, 18.0811920
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5334396, 26.5447769
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8766174, 34.8776093
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6929779, 27.6504364
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7445679, 18.7300949
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7571259, 22.7235107
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7512589, 28.7477570
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4645386, 25.4679718
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0087662, 17.0225601
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3173981, 17.3423042
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0928040, 19.0990257
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7612152, 18.7787437
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4386520, 19.4620285
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5940781, 20.6043320
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2569427, 20.2776413
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3630981, 27.3964844
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0450516, 22.0757065
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4683304, 17.4765015
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1515427, 24.1586456
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6857529, 18.6871185
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1071167, 23.1164017
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0533295, 32.0877228
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5850906, 24.6067276
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4021149, 24.4290848
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8602295, 28.8813705
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7409973, 33.7658081
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5236969, 37.5491638
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2784729, 39.2838440
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7881927, 24.7881622
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5281219, 24.5389328
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8454208, 15.8647652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1777

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2023611, upper bound: 9.0903758
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2044973, upper bound: 9.0881116
time: 26.52 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 38.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 38.35
Output dim: 2, lower bound: -9.0881116, upper bound: 9.2044973
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 38.35
Output dim: 2, lower bound: -9.0903758, upper bound: 9.2023611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 38.35
Output dim: 2, lower bound: -9.2023611, upper bound: 9.0903758
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 38.35
Output dim: 2, lower bound: -9.2044973, upper bound: 9.0881116

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8331604, 21.8539734
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0593452, 14.0887260
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3465195, 15.3599396
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6512337, 16.6608734
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3163033, 18.3436852
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4344635, 18.4555740
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0127029, 21.0139122
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2694550, 16.3017426
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5133133, 20.5452003
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.4851074, 21.5035248
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3005219, 27.3207550
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0790596, 18.0820045
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5424194, 26.5306091
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8782196, 34.8722076
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6450577, 27.6915817
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7306328, 18.7432137
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7193832, 22.7575302
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7475662, 28.7510757
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4675369, 25.4656906
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0221863, 17.0109825
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3420563, 17.3185539
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0979156, 19.0969429
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7783966, 18.7608528
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4619217, 19.4385262
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6034317, 20.5956650
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2756882, 20.2582436
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3945541, 27.3603897
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0756683, 22.0450134
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4764252, 17.4682083
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1562424, 24.1530762
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6864319, 18.6905556
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1165314, 23.1037064
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0876923, 32.0496216
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6069870, 24.5770416
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4294052, 24.3947906
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8816681, 28.8538589
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7645721, 33.7393646
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5493622, 37.5189514
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2826080, 39.2770996
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7879944, 24.7880936
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5386963, 24.5240555
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8653717, 15.8450356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.0848970, upper bound: 9.1878149
time: 25.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.0777155, upper bound: 9.2026621
time: 30.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8357468, 21.8527145
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0627632, 14.0850792
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3462906, 15.3600807
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6488609, 16.6620712
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3173790, 18.3410339
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4347458, 18.4555206
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0129623, 21.0147133
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2747345, 16.2965012
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5174332, 20.5409012
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.4863434, 21.5026703
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3005371, 27.3207626
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0811958, 18.0788689
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5419464, 26.5334396
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8731995, 34.8766174
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6504364, 27.6876068
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7287407, 18.7445679
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7235107, 22.7530060
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7477493, 28.7510681
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4679718, 25.4641113
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0225601, 17.0083885
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3423004, 17.3171463
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0990295, 19.0916939
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7783661, 18.7612190
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4620285, 19.4385452
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6043320, 20.5931778
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2776413, 20.2549973
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3937607, 27.3631058
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0756683, 22.0450439
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4763718, 17.4683342
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1586456, 24.1491394
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6871262, 18.6850700
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1129913, 23.1071167
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0840149, 32.0533295
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5986862, 24.5850830
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4217758, 24.4021072
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8750153, 28.8602219
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7641602, 33.7409973
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5444183, 37.5236969
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2824860, 39.2784729
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7880707, 24.7881851
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5348663, 24.5281296
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8643799, 15.8454247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.0871652, upper bound: 9.1856750
time: 26.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.0799826, upper bound: 9.2005225
time: 27.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8527145, 21.8341675
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0850792, 14.0629921
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3601685, 15.3462944
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6632423, 16.6488571
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3410301, 18.3190842
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4555206, 18.4345093
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0136185, 21.0129585
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2965012, 16.2746925
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5409012, 20.5176163
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5026703, 21.4859619
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3207550, 27.3005295
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0788765, 18.0821877
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5310898, 26.5419464
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8772278, 34.8731995
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6876144, 27.6493835
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7451057, 18.7287407
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7530060, 22.7239075
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7510605, 28.7475739
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4641113, 25.4691162
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0083923, 17.0247726
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3171387, 17.3433800
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0916901, 19.1031685
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7608337, 18.7783699
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4385452, 19.4619102
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5931778, 20.6059113
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2549973, 20.2789841
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3610458, 27.3937683
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0450134, 22.0756683
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4682617, 17.4763756
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1491394, 24.1601791
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6850662, 18.6919174
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1072388, 23.1129913
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0532990, 32.0840149
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5853500, 24.5986938
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4024277, 24.4217682
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8605194, 28.8750229
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7397003, 33.7641602
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5239105, 37.5444183
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2772369, 39.2824707
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7880402, 24.7880630
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5280151, 24.5348663
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8460464, 15.8643761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2005225, upper bound: 9.0799826
time: 54.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1856750, upper bound: 9.0871652
time: 28.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8553009, 21.8331528
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0884972, 14.0593491
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3599396, 15.3464355
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6608772, 16.6500549
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3421059, 18.3163033
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4557953, 18.4344673
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0139084, 21.0137596
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3017883, 16.2694511
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5450211, 20.5133171
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5039062, 21.4851074
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3207550, 27.3005371
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0810051, 18.0790558
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5306091, 26.5447769
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8722076, 34.8776093
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6929779, 27.6450653
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7432137, 18.7300949
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7571259, 22.7193832
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7512589, 28.7475586
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4645386, 25.4675369
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0087662, 17.0221825
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3173981, 17.3420525
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0928040, 19.0979195
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7608490, 18.7787437
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4386520, 19.4619217
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5940781, 20.6034317
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2569427, 20.2756882
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3603897, 27.3964844
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0450134, 22.0757065
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4682083, 17.4765015
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1515427, 24.1562424
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6857529, 18.6864319
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1037064, 23.1164017
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0496216, 32.0877228
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5770493, 24.6067276
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.3947983, 24.4290848
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8538666, 28.8813705
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7393646, 33.7658081
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5189514, 37.5491638
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2771149, 39.2838440
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7881012, 24.7881622
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5240479, 24.5389328
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8450394, 15.8647652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2026621, upper bound: 9.0777155
time: 35.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1878149, upper bound: 9.0848970
time: 27.63 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 65.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.0848970, upper bound: 9.1878149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.0777155, upper bound: 9.2026621
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.0871652, upper bound: 9.1856750
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.0799826, upper bound: 9.2005225
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.2005225, upper bound: 9.0799826
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.1856750, upper bound: 9.0871652
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.2026621, upper bound: 9.0777155
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 65.55
Output dim: 2, lower bound: -9.1878149, upper bound: 9.0848970

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8318710, 21.8539734
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0581055, 14.0887260
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3435135, 15.3599396
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6495438, 16.6608734
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3140373, 18.3436852
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4299011, 18.4555740
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0117416, 21.0139122
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.2665749, 16.3017426
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5102005, 20.5452003
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.4832230, 21.5035248
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.2996368, 27.3207550
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0790596, 18.0805473
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5424194, 26.5291061
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8774414, 34.8722076
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6450577, 27.6905060
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7290115, 18.7432137
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7186432, 22.7575302
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7475662, 28.7491074
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4675369, 25.4610519
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0221863, 17.0106506
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3420563, 17.3176155
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0979156, 19.0953751
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7783966, 18.7589912
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4619217, 19.4362106
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.6034317, 20.5941544
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2756882, 20.2559166
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3945541, 27.3543320
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0756683, 22.0419617
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4763870, 17.4682083
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1562424, 24.1510773
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6861572, 18.6905556
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1165314, 23.1036453
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0876923, 32.0493317
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.6069870, 24.5729752
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.4294052, 24.3921661
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8816681, 28.8522720
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7645721, 33.7358398
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5493622, 37.5151062
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2818909, 39.2770996
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7879944, 24.7880554
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5386505, 24.5240555
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8653717, 15.8450165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1785

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.0747316, upper bound: 9.1827685
time: 24.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.0662869, upper bound: 9.2002236
time: 32.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.4515257, 17.7626877, -7.4515257, 17.7626877, -21.8564606, 21.8318710
1: -0.8167844, 19.0394402, -0.8167844, 19.0394402, -14.0886078, 14.0581055
2: -1.2630103, 19.0099449, -1.2630103, 19.0099449, -15.3603668, 15.3434296
3: -7.5810738, 15.5562181, -7.5810738, 15.5562181, -16.6624146, 16.6483688
4: -4.9544535, 17.5765572, -4.9544535, 17.5765572, -18.3423424, 18.3140373
5: -6.5357847, 18.0377121, -6.5357847, 18.0377121, -18.4563446, 18.4298973
6: -32.9744263, -5.5525475, -32.9744263, -5.5525475, -21.0149002, 21.0127983
7: -5.7621326, 18.0761490, -5.7621326, 18.0761490, -16.3020744, 16.2665710
8: -16.5319309, 8.8552361, -16.5319309, 8.8552361, -20.5453415, 20.5102081
9: -5.6717424, 19.6510658, -5.6717424, 19.6510658, -21.5041428, 21.4832306
10: -21.8859138, 13.0249491, -21.8859138, 13.0249491, -27.3204498, 27.2996597
11: -16.6970882, 5.9592533, -16.6970882, 5.9592533, -18.0795479, 18.0790062
12: -35.2564545, -3.5988255, -35.2564545, -3.5988255, -26.5291061, 26.5489120
13: -21.3128242, 16.2911873, -21.3128242, 16.2911873, -34.8730927, 34.8768311
14: -52.7803230, -14.7402515, -52.7803230, -14.7402515, -27.6919098, 27.6453400
15: -13.0274439, 11.8806686, -13.0274439, 11.8806686, -18.7434769, 18.7284737
16: -13.0913773, 16.1038418, -13.0913773, 16.1038418, -22.7598419, 22.7186432
17: -46.4519310, -12.0110188, -46.4519310, -12.0110188, -28.7492828, 28.7477646
18: -17.1505623, 11.4658041, -17.1505623, 11.4658041, -25.4598923, 25.4680099
19: -16.7601261, 6.1205573, -16.7601261, 6.1205573, -17.0084305, 17.0220413
20: -19.5932732, 2.8888607, -19.5932732, 2.8888607, -17.3164673, 17.3439903
21: -19.4045105, 5.4689670, -19.4045105, 5.4689670, -19.0912323, 19.0983963
22: -19.9513493, 6.1021409, -19.9513493, 6.1021409, -18.7589874, 18.7789345
23: -15.8755713, 9.2936087, -15.8755713, 9.2936087, -19.4363403, 19.4621506
24: -14.6717510, 9.7053413, -14.6717510, 9.7053413, -20.5925674, 20.6035843
25: -19.4776974, 5.8866720, -19.4776974, 5.8866720, -20.2546158, 20.2759247
26: -33.1973572, 4.2804880, -33.1973572, 4.2804880, -27.3543320, 27.3970947
27: -16.4157562, 11.1851311, -16.4157562, 11.1851311, -27.6008873, 27.6008873
28: -18.5409164, 9.6061411, -18.5409164, 9.6061411, -22.0419617, 22.0760345
29: -17.3990765, 3.6752372, -17.3990765, 3.6752372, -17.4686508, 17.4764595
30: -24.8458824, 1.7533870, -24.8458824, 1.7533870, -24.1495361, 24.1563873
31: -17.8102646, 6.6330538, -17.8102646, 6.6330538, -18.6861115, 18.6861572
32: -24.0971680, 2.6651855, -24.0971680, 2.6651855, -23.1036453, 23.1169510
33: -40.9145126, -2.0184083, -40.9145126, -2.0184083, -32.0493317, 32.0885468
34: -40.1440811, -9.4770069, -40.1440811, -9.4770069, -24.5729752, 24.6071396
35: -30.2203903, 0.4602835, -30.2203903, 0.4602835, -24.3921738, 24.4293518
36: -30.1097603, 4.7512007, -30.1097603, 4.7512007, -28.8522644, 28.8815536
37: -47.5639839, -6.8119135, -47.5639839, -6.8119135, -33.7358398, 33.7667389
38: -38.1440659, 4.7428694, -38.1440659, 4.7428694, -37.5151062, 37.5501251
39: -42.1536865, -1.3208117, -42.1536865, -1.3208117, -39.2772522, 39.2830811
40: -39.7446480, -10.9497890, -39.7446480, -10.9497890, -24.7880707, 24.7882156
41: -22.8790722, 6.2636433, -22.8790722, 6.2636433, -24.5240326, 24.5389023
42: -24.6491756, -2.2389681, -24.6491756, -2.2389681, -15.8450165, 15.8648643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=195, inp2_unstable=195, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=154, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1516

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1785

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2002236, upper bound: 9.0662869
time: 27.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1827685, upper bound: 9.0747316
time: 36.44 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 66.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 66.19
Output dim: 2, lower bound: -9.0747316, upper bound: 9.1827685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 66.19
Output dim: 2, lower bound: -9.0662869, upper bound: 9.2002236
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 66.19
Output dim: 2, lower bound: -9.2002236, upper bound: 9.0662869
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 66.19
Output dim: 2, lower bound: -9.1827685, upper bound: 9.0747316

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 43.22 + 1456.00 = 1499.22 seconds
