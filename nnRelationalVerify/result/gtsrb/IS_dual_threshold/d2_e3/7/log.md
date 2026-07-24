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
execution time: IAR + RelationalAnalysis = 2.55 + 41.13 = 43.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -9.2114816, upper bound: 9.2114816

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2050112, upper bound: 9.1839168
time: 9.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2050112, upper bound: 9.2050110
time: 27.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 37.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 37.54
Output dim: 2, lower bound: -9.2050112, upper bound: 9.1839168
IS_A2, status: Status.UNKNOWN, split count: 1, time: 37.54
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

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1848230, upper bound: 9.1822258
time: 31.44 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2042147, upper bound: 9.1831232
time: 28.10 seconds

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

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1398

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1848230, upper bound: 9.2032825
time: 26.96 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2042147, upper bound: 9.2042148
time: 28.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 57.70 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 57.70
Output dim: 2, lower bound: -9.1848230, upper bound: 9.1822258
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 57.70
Output dim: 2, lower bound: -9.2042147, upper bound: 9.1831232
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 57.70
Output dim: 2, lower bound: -9.1848230, upper bound: 9.2032825
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 57.70
Output dim: 2, lower bound: -9.2042147, upper bound: 9.2042148

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -7.4430008, 17.7596340, -7.4476767, 17.7613640, -21.8933105, 21.8327255
1: -0.8108716, 19.0367146, -0.8140333, 19.0382385, -14.1343765, 14.1244164
2: -1.2580884, 18.9820728, -1.2606418, 18.9984283, -15.3407707, 15.3498955
3: -7.5735283, 15.5512943, -7.5777273, 15.5540695, -16.6591873, 16.7241249
4: -4.9457045, 17.5603542, -4.9504447, 17.5698776, -18.3609161, 18.3776245
5: -6.5287390, 18.0287285, -6.5326967, 18.0339031, -18.4604492, 18.5020714
6: -32.9244423, -5.5576372, -32.9537659, -5.5547552, -20.9718704, 20.8963623
7: -5.7538891, 18.0723000, -5.7581787, 18.0745277, -16.3253708, 16.2727737
8: -16.5255089, 8.8367329, -16.5288124, 8.8475504, -20.5922508, 20.6138687
9: -5.6610651, 19.6469955, -5.6669564, 19.6492996, -21.5595322, 21.6322632
10: -21.8770428, 13.0025482, -21.8818493, 13.0156555, -27.3621750, 27.3748016
11: -16.6910706, 5.9503074, -16.6944771, 5.9551973, -18.0769272, 18.0907135
12: -35.2395325, -3.6126251, -35.2489891, -3.6050487, -26.5036774, 26.5610733
13: -21.3030987, 16.2834759, -21.3085022, 16.2879124, -34.8870392, 34.8876953
14: -52.7738800, -14.7993946, -52.7774849, -14.7647505, -27.5095520, 27.8370056
15: -13.0196705, 11.8774166, -13.0239305, 11.8792658, -18.7327614, 18.7907753
16: -13.0820150, 16.0985603, -13.0868969, 16.1014671, -22.8093109, 22.9168777
17: -46.4439850, -12.0756969, -46.4484024, -12.0378857, -28.4190369, 28.7015076
18: -17.1439285, 11.4552412, -17.1476479, 11.4611874, -25.5649109, 25.4211197
19: -16.7543831, 6.1161880, -16.7577057, 6.1185393, -17.0688477, 17.0206604
20: -19.5899811, 2.8837914, -19.5918655, 2.8865764, -17.3702927, 17.3533173
21: -19.3988152, 5.4612503, -19.4020538, 5.4654303, -19.0902863, 19.0726852
22: -19.9423656, 6.0967760, -19.9475651, 6.0997019, -18.7504654, 18.7806740
23: -15.8717785, 9.2861443, -15.8739796, 9.2901793, -19.5273361, 19.4653893
24: -14.6677771, 9.6874809, -14.6699886, 9.6977167, -20.6443558, 20.5860672
25: -19.4743614, 5.8677201, -19.4762344, 5.8784966, -20.2896500, 20.2628670
26: -33.1901321, 4.2733774, -33.1942215, 4.2773557, -27.4572449, 27.4014587
27: -16.3875618, 11.1786423, -16.4041271, 11.1819382, -27.5695000, 27.5827694
28: -18.5305195, 9.5987701, -18.5365753, 9.6027908, -22.1246719, 22.1020126
29: -17.3872128, 3.6705310, -17.3940487, 3.6730800, -17.4578018, 17.4638824
30: -24.8434448, 1.7447119, -24.8448486, 1.7495441, -24.1050949, 24.1291428
31: -17.8064728, 6.6168804, -17.8086300, 6.6257639, -18.6944122, 18.6648827
32: -24.0460472, 2.6595421, -24.0759583, 2.6627913, -23.0545731, 23.0852814
33: -40.9085655, -2.0268869, -40.9119339, -2.0222235, -32.1828461, 32.1888275
34: -40.1094589, -9.4845810, -40.1294327, -9.4805927, -24.6365356, 24.6452255
35: -30.2061634, 0.4550052, -30.2144279, 0.4574542, -24.5359192, 24.4996872
36: -30.0578651, 4.7436991, -30.0883598, 4.7476759, -28.9041748, 28.9223862
37: -47.5314560, -6.8243842, -47.5502586, -6.8177557, -33.8089905, 33.8055420
38: -38.0965309, 4.7346487, -38.1243248, 4.7391105, -37.5936432, 37.5707397
39: -42.1423798, -1.3257184, -42.1487045, -1.3228998, -39.2993164, 39.3032227
40: -39.7110558, -10.9548264, -39.7304459, -10.9519529, -24.7428436, 24.7539062
41: -22.8171883, 6.2562342, -22.8535938, 6.2603340, -24.5274200, 24.4566803
42: -24.6099720, -2.2462404, -24.6327972, -2.2423167, -15.7896805, 15.8103485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=153, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1727986, upper bound: 9.1818835
time: 25.97 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2029990, upper bound: 9.1818835
time: 38.74 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -7.4397373, 17.7650452, -7.4072695, 17.7605247, -21.8880920, 21.8820038
1: -0.7933123, 19.0227547, -0.7826807, 19.0376968, -14.1475983, 14.1327133
2: -1.3186920, 18.9876785, -1.2328038, 19.0062828, -15.4339294, 15.3232651
3: -7.5861607, 15.5261755, -7.5448208, 15.5528603, -16.6956635, 16.6576157
4: -4.9787598, 17.5494995, -4.9084911, 17.5735474, -18.3961639, 18.3368530
5: -6.5413485, 17.9966469, -6.5004568, 18.0304317, -18.4947739, 18.4521255
6: -32.9570656, -5.4267368, -32.9585037, -5.5620379, -20.9979172, 21.1420593
7: -5.7332649, 18.0443382, -5.7159481, 18.0739708, -16.3475990, 16.3232880
8: -16.5470848, 8.8351154, -16.4868507, 8.8513756, -20.6454239, 20.5940361
9: -5.6172123, 19.6670170, -5.6308336, 19.6486282, -21.5483780, 21.5396576
10: -21.9225140, 13.0228643, -21.8654766, 13.0146074, -27.4902344, 27.3661957
11: -16.7137089, 5.9844542, -16.6918983, 5.9428253, -18.0960007, 18.1032181
12: -35.2564392, -3.5607977, -35.2510757, -3.6227131, -26.5520096, 26.5494080
13: -21.3124275, 16.2500763, -21.2851963, 16.2850380, -34.8997345, 34.8369751
14: -52.8998489, -14.7535715, -52.7711754, -14.7520552, -27.9931488, 27.8332596
15: -13.0819654, 11.8754463, -13.0139999, 11.8757687, -18.8491631, 18.7785416
16: -13.0538101, 16.1213017, -13.0557661, 16.1012611, -22.8228607, 22.8301926
17: -46.6221161, -12.0513058, -46.4433136, -12.0319252, -28.9394836, 28.6811600
18: -17.1360168, 11.4354630, -17.1436195, 11.4361334, -25.3994751, 25.4350891
19: -16.7432690, 6.0870333, -16.7555523, 6.0915518, -17.0099640, 17.0564079
20: -19.5715733, 2.8756132, -19.5899696, 2.8684320, -17.3527069, 17.3691673
21: -19.3948975, 5.4577198, -19.3994904, 5.4416342, -19.0569992, 19.1071472
22: -19.9482498, 6.0519571, -19.9469624, 6.0691166, -18.7928085, 18.7830200
23: -15.8380013, 9.2384434, -15.8723984, 9.2573328, -19.4592743, 19.4776230
24: -14.6613140, 9.6564941, -14.6681442, 9.6687164, -20.5933151, 20.5867157
25: -19.4716301, 5.8365870, -19.4739914, 5.8440866, -20.2961426, 20.2781792
26: -33.1637993, 4.2141657, -33.1920471, 4.2368689, -27.3932877, 27.4039993
27: -16.3898582, 11.2189322, -16.4060001, 11.1602011, -27.5500603, 27.6249313
28: -18.4940205, 9.5760059, -18.5342598, 9.5671387, -22.0868835, 22.1338348
29: -17.3875504, 3.6575534, -17.3928394, 3.6428447, -17.4709930, 17.4836502
30: -24.8337193, 1.7588801, -24.8435402, 1.7319713, -24.1499786, 24.1345978
31: -17.8117161, 6.5898266, -17.8045540, 6.5980120, -18.6499557, 18.6780434
32: -24.0763397, 2.7553558, -24.0779686, 2.6559620, -23.0842514, 23.2014084
33: -40.8976135, -1.9856558, -40.9064445, -2.0290828, -32.1827850, 32.2214508
34: -40.1323128, -9.4178352, -40.1358185, -9.4925289, -24.6259842, 24.7052841
35: -30.2032585, 0.4693213, -30.2127056, 0.4413352, -24.4879837, 24.5346069
36: -30.1018353, 4.8276706, -30.1017361, 4.7310367, -28.9184570, 29.0264969
37: -47.5408478, -6.7216320, -47.5514412, -6.8263993, -33.8116608, 33.9226685
38: -38.1191750, 4.7985792, -38.1326218, 4.7115984, -37.5613098, 37.6732330
39: -42.1358566, -1.2834797, -42.1402054, -1.3270950, -39.2836304, 39.3627167
40: -39.6945686, -10.8507195, -39.7182159, -10.9565878, -24.7374268, 24.8664551
41: -22.8765373, 6.3888392, -22.8689327, 6.2546473, -24.5686188, 24.7123642
42: -24.6527214, -2.1370616, -24.6452255, -2.2479725, -15.8200378, 15.9342651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1398

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1619

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1689436, upper bound: 9.2021503
time: 28.22 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1836979, upper bound: 9.2021503
time: 32.04 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -7.5274181, 17.7755089, -7.4478426, 17.7617245, -21.9738731, 21.9349976
1: -0.8598380, 19.0468731, -0.8143775, 19.0384026, -14.1831665, 14.1915741
2: -1.3747585, 19.0111599, -1.2606819, 19.0072384, -15.4645500, 15.3778801
3: -7.6528163, 15.5606661, -7.5785236, 15.5543938, -16.7343140, 16.6983337
4: -5.0661311, 17.5764809, -4.9509301, 17.5742569, -18.4379425, 18.4062080
5: -6.6060681, 18.0346050, -6.5327721, 18.0326920, -18.5221024, 18.5059319
6: -32.9803848, -5.4047203, -32.9694939, -5.5545492, -21.0226288, 21.1762619
7: -5.8178177, 18.0796165, -5.7583551, 18.0750847, -16.3817673, 16.4078751
8: -16.6337318, 8.8622932, -16.5291290, 8.8528404, -20.7026749, 20.6562843
9: -5.6871262, 19.6936493, -5.6628208, 19.6493683, -21.6052246, 21.6022568
10: -21.9630928, 13.0396633, -21.8831005, 13.0179911, -27.5334702, 27.4027252
11: -16.7340870, 6.0142717, -16.6947575, 5.9561815, -18.1353912, 18.1296082
12: -35.2615967, -3.5187411, -35.2528191, -3.6049080, -26.5644302, 26.6260071
13: -21.3632965, 16.3065720, -21.3093948, 16.2888279, -34.9546661, 34.9198914
14: -52.9168358, -14.7389860, -52.7773323, -14.7463388, -28.0146484, 27.8883133
15: -13.1084251, 11.8835373, -13.0247879, 11.8794956, -18.8959579, 18.7972221
16: -13.1192970, 16.1523056, -13.0853043, 16.1018028, -22.8523026, 22.8988113
17: -46.6330185, -12.0143518, -46.4487457, -12.0178204, -28.9491577, 28.7449112
18: -17.1819859, 11.4863653, -17.1472549, 11.4606647, -25.4706802, 25.4700470
19: -16.7841644, 6.1405325, -16.7584114, 6.1186800, -17.0808105, 17.0720139
20: -19.5963917, 2.9136209, -19.5917110, 2.8870656, -17.3993950, 17.3809929
21: -19.4279480, 5.5082846, -19.4025879, 5.4665661, -19.1173401, 19.1339417
22: -19.9894104, 6.1122484, -19.9494591, 6.1000147, -18.8703918, 18.8076973
23: -15.8909836, 9.3072338, -15.8741074, 9.2908077, -19.5508575, 19.4908676
24: -14.7059612, 9.7138519, -14.6699429, 9.6970062, -20.6688538, 20.6149445
25: -19.5146389, 5.9078884, -19.4760361, 5.8792996, -20.3799210, 20.3031387
26: -33.2193146, 4.2946281, -33.1950874, 4.2778692, -27.4925766, 27.4291229
27: -16.4310760, 11.2656841, -16.4095154, 11.1824875, -27.6135635, 27.6751995
28: -18.5454903, 9.6476278, -18.5367546, 9.6031561, -22.1790314, 22.1527405
29: -17.4287052, 3.7183969, -17.3952694, 3.6729519, -17.5469666, 17.5117340
30: -24.8547363, 1.7994528, -24.8447266, 1.7505288, -24.1893692, 24.1660080
31: -17.8520622, 6.6484022, -17.8081017, 6.6266747, -18.7220306, 18.7182503
32: -24.1062851, 2.7769957, -24.0918236, 2.6626024, -23.1111755, 23.2349014
33: -40.9177475, -1.9601851, -40.9118156, -2.0212917, -32.2073975, 32.2655792
34: -40.1487045, -9.3937626, -40.1387520, -9.4809065, -24.6635590, 24.7387772
35: -30.2262154, 0.5085721, -30.2166672, 0.4576883, -24.5307541, 24.5641861
36: -30.1210060, 4.8645020, -30.1050186, 4.7475648, -28.9606476, 29.0638580
37: -47.5667725, -6.7015724, -47.5572968, -6.8173752, -33.8389282, 33.9607391
38: -38.1553040, 4.8551250, -38.1375542, 4.7389321, -37.6281738, 37.7153625
39: -42.1619415, -1.2614174, -42.1494904, -1.3227272, -39.3143158, 39.3977203
40: -39.7387924, -10.8220406, -39.7365646, -10.9519520, -24.7687759, 24.9149399
41: -22.8904457, 6.4002609, -22.8734837, 6.2597666, -24.5908661, 24.7316666
42: -24.6563396, -2.1244450, -24.6467171, -2.2426775, -15.8320694, 15.9534073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=195, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1398

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1619

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.1883308, upper bound: 9.2030989
time: 25.57 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -9.2030988, upper bound: 9.2030989
time: 27.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 55.50 seconds
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.1727986, upper bound: 9.1818835
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.2029990, upper bound: 9.1818835
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.1689436, upper bound: 9.2021503
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.1836979, upper bound: 9.2021503
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.1883308, upper bound: 9.2030989
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 55.50
Output dim: 2, lower bound: -9.2030988, upper bound: 9.2030989

## BFS IS instance: IS_A1_A2_B2

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

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1740

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1619

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1871131, upper bound: 9.1809293
time: 35.47 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2018856, upper bound: 9.1809293
time: 28.62 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -7.5230770, 17.7623825, -7.4391327, 17.7350845, -21.9432411, 21.9139862
1: -0.8580430, 19.0416374, -0.8108032, 19.0277214, -14.1683121, 14.1819572
2: -1.3730223, 18.9962292, -1.2572327, 18.9774933, -15.4344864, 15.3593788
3: -7.6503620, 15.5557461, -7.5737419, 15.5444365, -16.7219276, 16.6876030
4: -5.0637789, 17.5706291, -4.9462795, 17.5626602, -18.4267044, 18.3967285
5: -6.6026368, 18.0239105, -6.5259771, 18.0113583, -18.5026741, 18.4899750
6: -32.9668007, -5.4076734, -32.9421654, -5.5605154, -21.0023727, 21.1440887
7: -5.8147883, 18.0780334, -5.7523508, 18.0719166, -16.3753662, 16.3995590
8: -16.6324959, 8.8606930, -16.5267143, 8.8496218, -20.6964645, 20.6510544
9: -5.6835232, 19.6811352, -5.6556320, 19.6238480, -21.5738068, 21.5824356
10: -21.9617233, 13.0175037, -21.8803520, 12.9732637, -27.4842529, 27.3759003
11: -16.7179317, 6.0127468, -16.6618347, 5.9530945, -18.1160851, 18.0946426
12: -35.2605515, -3.5355935, -35.2506943, -3.6390567, -26.5250015, 26.6047058
13: -21.3589573, 16.2557125, -21.3005676, 16.1855450, -34.8470306, 34.8601532
14: -52.9121475, -14.7594414, -52.7679329, -14.7877178, -27.9663162, 27.8573227
15: -13.1061954, 11.8703852, -13.0203676, 11.8531170, -18.8719635, 18.7810211
16: -13.1141396, 16.1452122, -13.0749197, 16.0875950, -22.8315430, 22.8805313
17: -46.6310120, -12.0315742, -46.4446716, -12.0527363, -28.9102173, 28.7226028
18: -17.1603355, 11.4842243, -17.1032829, 11.4563723, -25.4444275, 25.4233627
19: -16.7673397, 6.1399946, -16.7252617, 6.1177039, -17.0620651, 17.0364609
20: -19.5674515, 2.9119642, -19.5329113, 2.8837972, -17.3672447, 17.3197784
21: -19.4017143, 5.5074492, -19.3491745, 5.4649272, -19.0889816, 19.0787201
22: -19.9730492, 6.1109452, -19.9162579, 6.0973482, -18.8512726, 18.7730942
23: -15.8712807, 9.3057690, -15.8345432, 9.2879601, -19.5283508, 19.4488792
24: -14.6867523, 9.7122993, -14.6310654, 9.6941290, -20.6467209, 20.5747452
25: -19.4985790, 5.9061666, -19.4432373, 5.8759279, -20.3627472, 20.2748032
26: -33.1927414, 4.2907586, -33.1410599, 4.2700672, -27.4585953, 27.3715515
27: -16.3949528, 11.2636280, -16.3362007, 11.1783552, -27.5733070, 27.5998287
28: -18.5141659, 9.6464825, -18.4741173, 9.6008301, -22.1451263, 22.0879135
29: -17.4181595, 3.7168894, -17.3740959, 3.6698947, -17.5329590, 17.4881096
30: -24.8241692, 1.7977390, -24.7824936, 1.7472391, -24.1553726, 24.1019135
31: -17.8343029, 6.6471872, -17.7719536, 6.6243105, -18.7017059, 18.6801796
32: -24.0999718, 2.7728367, -24.0792427, 2.6541748, -23.0967636, 23.2191010
33: -40.9118195, -1.9628744, -40.8998413, -2.0265927, -32.1910095, 32.2470703
34: -40.1289253, -9.3954430, -40.0990410, -9.4842968, -24.6390533, 24.6943207
35: -30.2213173, 0.5080364, -30.2067986, 0.4565759, -24.5181351, 24.5449677
36: -30.1149483, 4.8637657, -30.0932541, 4.7462158, -28.9428101, 29.0387192
37: -47.5612411, -6.7035570, -47.5460815, -6.8213925, -33.8289032, 33.9470139
38: -38.1468544, 4.8522935, -38.1203613, 4.7332850, -37.6086121, 37.6878510
39: -42.1557388, -1.2822504, -42.1369896, -1.3641968, -39.2654266, 39.3643036
40: -39.7292709, -10.8245649, -39.7173195, -10.9570789, -24.7550049, 24.8957748
41: -22.8821564, 6.3972383, -22.8569946, 6.2536998, -24.5726471, 24.7053223
42: -24.6471882, -2.1279709, -24.6283188, -2.2497981, -15.8176308, 15.9362602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=154, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1398

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1569111, upper bound: 9.2018858
time: 35.98 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1871131, upper bound: 9.2018858
time: 33.12 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.5269098, 17.7742424, -7.5228615, 17.7635803, -21.9684143, 22.0100327
1: -0.8595822, 19.0452499, -0.8525715, 19.0384064, -14.1835022, 14.2325134
2: -1.3745527, 19.0103836, -1.3175094, 19.0100193, -15.4554214, 15.4337959
3: -7.6524901, 15.5594358, -7.6206608, 15.5568647, -16.7429123, 16.7360458
4: -5.0657759, 17.5760384, -5.0068111, 17.5767708, -18.4399872, 18.4567108
5: -6.6057625, 18.0335960, -6.5753918, 18.0363312, -18.5268860, 18.5446472
6: -32.9793663, -5.4051151, -32.9756317, -5.4977274, -21.0820312, 21.1790695
7: -5.8173771, 18.0794563, -5.7811332, 18.0814018, -16.3909302, 16.4360886
8: -16.6335449, 8.8618927, -16.5659771, 8.8591547, -20.7163467, 20.6908951
9: -5.6867113, 19.6925354, -5.7110071, 19.6533298, -21.6020889, 21.6539459
10: -21.9627075, 13.0386105, -21.9476738, 13.0313540, -27.5403748, 27.4682617
11: -16.7323799, 6.0141478, -16.7077370, 6.0103703, -18.1911774, 18.1330147
12: -35.2613983, -3.5202880, -35.2862549, -3.5814347, -26.5780945, 26.6658401
13: -21.3625183, 16.3021851, -21.4298801, 16.2924900, -34.9423218, 35.0357666
14: -52.9162560, -14.7402554, -52.8426857, -14.7401505, -28.0096664, 27.9558411
15: -13.1081104, 11.8823528, -13.0917664, 11.8835449, -18.8914719, 18.8635101
16: -13.1185503, 16.1511269, -13.1105652, 16.1034355, -22.8532944, 22.9248962
17: -46.6328545, -12.0163364, -46.5175171, -12.0099154, -28.9395981, 28.8202667
18: -17.1797676, 11.4860630, -17.1569748, 11.5227528, -25.5308304, 25.4730453
19: -16.7829514, 6.1404104, -16.7681122, 6.1656971, -17.1324005, 17.0718651
20: -19.5936852, 2.9134076, -19.5941238, 2.9541438, -17.4656143, 17.3684196
21: -19.4254971, 5.5081453, -19.4131508, 5.5418448, -19.1979599, 19.1290245
22: -19.9880104, 6.1119957, -19.9561958, 6.1368985, -18.9069061, 18.8100891
23: -15.8891392, 9.3069630, -15.8784523, 9.3369637, -19.5969543, 19.4875412
24: -14.7040110, 9.7137280, -14.6747189, 9.7479210, -20.7221909, 20.6109314
25: -19.5129566, 5.9076710, -19.4815216, 5.9314327, -20.4265060, 20.3027916
26: -33.2169800, 4.2939692, -33.2021332, 4.3318381, -27.5480804, 27.4283905
27: -16.4281673, 11.2653751, -16.4169884, 11.2739620, -27.7021294, 27.6823635
28: -18.5429134, 9.6472826, -18.5419769, 9.6752701, -22.2515106, 22.1455154
29: -17.4282913, 3.7180562, -17.4063244, 3.7032857, -17.5774078, 17.5223694
30: -24.8519020, 1.7991257, -24.8471718, 1.8352318, -24.2744751, 24.1489182
31: -17.8501396, 6.6483021, -17.8133583, 6.6793690, -18.7778473, 18.7130089
32: -24.1049500, 2.7764654, -24.0990524, 2.6929631, -23.1406631, 23.2422562
33: -40.9167061, -1.9605856, -40.9214630, -1.9992785, -32.2159271, 32.2892227
34: -40.1470146, -9.3941174, -40.1472206, -9.4334698, -24.7134705, 24.7395172
35: -30.2257347, 0.5082881, -30.2299690, 0.4683762, -24.5544128, 24.5643234
36: -30.1205540, 4.8642592, -30.1168442, 4.7563748, -28.9865112, 29.0598907
37: -47.5660095, -6.7018824, -47.5683746, -6.7932720, -33.8613434, 33.9715347
38: -38.1538162, 4.8547697, -38.1497574, 4.7603197, -37.6562958, 37.7173767
39: -42.1610947, -1.2633028, -42.1963234, -1.3063378, -39.3278046, 39.4436646
40: -39.7371368, -10.8227329, -39.7414856, -10.9179516, -24.7978821, 24.9156570
41: -22.8888206, 6.3997927, -22.8809013, 6.2922335, -24.6307678, 24.7324905
42: -24.6556873, -2.1250792, -24.6495705, -2.2038598, -15.8683281, 15.9561310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=193, inp2_unstable=194, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=155, inp2_unstable=155, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1740

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.1717046, upper bound: 9.2018858
time: 31.81 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -9.2018856, upper bound: 9.2018858
time: 28.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 62.40 seconds
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.1871131, upper bound: 9.1809293
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.2018856, upper bound: 9.1809293
IS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.1569111, upper bound: 9.2018858
IS_A2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.1871131, upper bound: 9.2018858
IS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.1717046, upper bound: 9.2018858
IS_A2_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 62.40
Output dim: 2, lower bound: -9.2018856, upper bound: 9.2018858

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 43.68 + 541.24 = 584.91 seconds
