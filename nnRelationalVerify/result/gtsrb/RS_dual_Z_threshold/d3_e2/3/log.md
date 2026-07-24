## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 7200 seconds
Split limit: 100
Threshold: 17.0611017154


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483)
1: (-14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343)
2: (-19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7155876, 19.7155876)
3: (-19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9897614, 19.9897652)
4: (-26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8187943, 24.8187943)
5: (-23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7634850, 21.7634926)
6: (-18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8734589, 24.8734589)
7: (-24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5915108, 22.5915146)
8: (-35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0472565, 30.0472565)
9: (-13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886)
10: (-12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454)
11: (-12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7116470, 25.7116432)
12: (-0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7497940, 30.7497978)
13: (-21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4417114, 39.4417114)
14: (-45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3931274, 34.3931274)
15: (-22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7730865, 20.7730865)
16: (-18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427)
17: (-34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2975159, 43.2975235)
18: (-11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266)
19: (-15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985)
20: (-10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372)
21: (-12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946)
22: (-13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147)
23: (-12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2109146, 19.2109146)
24: (-18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764)
25: (-10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917)
26: (-14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523)
27: (-24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310)
28: (-14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1722565, 23.1722603)
29: (-18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8146820, 26.8146896)
30: (-14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3655167, 27.3655167)
31: (-17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119)
32: (-13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9688110, 26.9688110)
33: (-32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0961914, 37.0961914)
34: (-25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1163635, 31.1163559)
35: (-19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6142273, 30.6142273)
36: (-19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0358734, 33.0358887)
37: (-28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.4023285, 33.4023209)
38: (-25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5146942, 41.5146942)
39: (-36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0180664, 44.0180740)
40: (-31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7548676, 27.7548637)
41: (-18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3267365, 26.3267403)
42: (-11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.88 + 90.66 = 93.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -17.0952923, upper bound: 17.0952923

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0909425, upper bound: 17.0349469
time: 36.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0349469, upper bound: 17.0909425
time: 37.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 73.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 73.64
Output dim: 12, lower bound: -17.0909425, upper bound: 17.0349469
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 73.64
Output dim: 12, lower bound: -17.0349469, upper bound: 17.0909425

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7106094, 19.7111778
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9793625, 19.9810028
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8127594, 24.8127823
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7432404, 21.7472153
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8772125, 24.8791275
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5725060, 22.5774956
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0325470, 30.0344620
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7100754, 25.7098427
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7330093, 30.7280922
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4361267, 39.4354782
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4045258, 34.4058304
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7659531, 20.7668800
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2950134, 43.2945557
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2100677, 19.2102242
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1735077, 23.1740990
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8152390, 26.8151855
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3668900, 27.3677139
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0966949, 37.0970535
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1196899, 31.1199951
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6197662, 30.6171341
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0289154, 33.0283737
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.4034348, 33.4037552
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5251923, 41.5224304
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0243530, 44.0229034
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7450867, 27.7472916
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3207703, 26.3210487
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0834439, upper bound: 16.9880285
time: 24.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0269031, upper bound: 17.0247820
time: 32.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7111740, 19.7106056
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9810028, 19.9793625
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8127823, 24.8127594
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7472229, 21.7432365
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8791199, 24.8772087
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5774956, 22.5725098
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0344620, 30.0325470
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7098465, 25.7100754
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7280884, 30.7330093
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4354706, 39.4361267
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4058228, 34.4045181
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7668839, 20.7659531
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2945557, 43.2950134
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2102203, 19.2100716
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1741028, 23.1735115
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8151855, 26.8152390
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3677063, 27.3668900
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0970612, 37.0967026
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1199951, 31.1196823
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6171417, 30.6197662
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0283813, 33.0289230
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.4037552, 33.4034348
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5224152, 41.5252075
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0229034, 44.0243530
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7472916, 27.7450867
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3210449, 26.3207664
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0247820, upper bound: 17.0269031
time: 33.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9880285, upper bound: 17.0834439
time: 33.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 69.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 69.56
Output dim: 12, lower bound: -17.0834439, upper bound: 16.9880285
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 69.56
Output dim: 12, lower bound: -17.0269031, upper bound: 17.0247820
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 69.56
Output dim: 12, lower bound: -17.0247820, upper bound: 17.0269031
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 69.56
Output dim: 12, lower bound: -16.9880285, upper bound: 17.0834439

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7104111, 19.7110138
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9772568, 19.9795837
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8127670, 24.8127670
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7345886, 21.7413063
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8755798, 24.8822365
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5634193, 22.5734062
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0273666, 30.0297470
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7100296, 25.7098007
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7252121, 30.7147217
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4347687, 39.4324036
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4031982, 34.4044571
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7630157, 20.7643509
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2914581, 43.2886505
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2096786, 19.2103729
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1731567, 23.1757545
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8151779, 26.8151398
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3661652, 27.3676682
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0948181, 37.0956955
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1192322, 31.1197128
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6203308, 30.6156540
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0273285, 33.0264435
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3988419, 33.4004593
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5260315, 41.5218811
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0276337, 44.0223541
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7341309, 27.7426300
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3193970, 26.3198662
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0329372, upper bound: 16.9791357
time: 45.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0745804, upper bound: 16.9375350
time: 37.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7110138, 19.7104111
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9795837, 19.9772606
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8127670, 24.8127670
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7413025, 21.7345848
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8822327, 24.8755836
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5734062, 22.5634155
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0297470, 30.0273666
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7098007, 25.7100334
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7147217, 30.7252159
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4324036, 39.4347687
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4044571, 34.4032059
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7643585, 20.7630119
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2886505, 43.2914581
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2103729, 19.2096786
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1757507, 23.1731491
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8151398, 26.8151817
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3676605, 27.3661652
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0957031, 37.0948181
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1197205, 31.1192245
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6156464, 30.6203308
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0264435, 33.0273209
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.4004593, 33.3988419
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5218964, 41.5260315
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0223541, 44.0276337
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7426300, 27.7341385
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3198700, 26.3194084
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9375350, upper bound: 17.0745804
time: 43.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9791357, upper bound: 17.0329372
time: 32.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 79.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 79.15
Output dim: 12, lower bound: -17.0329372, upper bound: 16.9791357
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 79.15
Output dim: 12, lower bound: -17.0745804, upper bound: 16.9375350
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 79.15
Output dim: 12, lower bound: -16.9375350, upper bound: 17.0745804
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 79.15
Output dim: 12, lower bound: -16.9791357, upper bound: 17.0329372

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6887321, 19.7023964
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9542160, 19.9714813
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8015137, 24.8072891
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.6992912, 21.7272797
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8753815, 24.8820877
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5396729, 22.5639725
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0149918, 30.0254669
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7107773, 25.7050743
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7248154, 30.6968002
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4284134, 39.4299316
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3572235, 34.3861771
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7615662, 20.7633400
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2780991, 43.2834473
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2093887, 19.2114906
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1725388, 23.1781731
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8158798, 26.8145638
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3641510, 27.3602753
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9668884, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0891571, 37.0814743
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1157990, 31.1086197
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6180573, 30.6122284
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0237885, 33.0202789
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3846207, 33.3646698
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5299225, 41.5175095
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0208588, 44.0101242
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7301331, 27.7319221
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3121414, 26.3017349
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0407054, upper bound: 16.9283495
time: 27.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0654034, upper bound: 16.9035709
time: 32.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7023888, 19.6887321
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9714813, 19.9542160
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8072891, 24.8015137
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7272758, 21.6992950
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8820953, 24.8753891
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5639725, 22.5396729
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0254669, 30.0149918
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7050781, 25.7107735
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.6968002, 30.7248192
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4299240, 39.4284210
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3861771, 34.3572235
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7633438, 20.7615700
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2834396, 43.2780991
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2114944, 19.2093925
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1781693, 23.1725349
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8145676, 26.8158798
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3602753, 27.3641510
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9668884
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0814667, 37.0891647
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1086121, 31.1157990
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6122284, 30.6180496
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0202789, 33.0237961
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3646774, 33.3846130
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5175018, 41.5299301
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0101166, 44.0208664
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7319183, 27.7301292
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3017349, 26.3121414
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9035709, upper bound: 17.0654034
time: 28.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9283495, upper bound: 17.0407054
time: 32.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 63.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 63.58
Output dim: 12, lower bound: -17.0407054, upper bound: 16.9283495
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 63.58
Output dim: 12, lower bound: -17.0654034, upper bound: 16.9035709
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 63.58
Output dim: 12, lower bound: -16.9035709, upper bound: 17.0654034
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 63.58
Output dim: 12, lower bound: -16.9283495, upper bound: 17.0407054

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6845894, 19.7023964
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9486771, 19.9714813
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.7954941, 24.8072891
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.6920929, 21.7272797
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8753815, 24.8812714
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5342064, 22.5639725
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0055084, 30.0254669
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7107773, 25.7009621
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7248154, 30.6964874
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4284134, 39.4295425
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3383179, 34.3861771
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7577362, 20.7633400
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2720871, 43.2834473
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2093887, 19.2080269
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1725388, 23.1747093
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8154221, 26.8145638
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3641510, 27.3575821
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9656754, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0891571, 37.0764008
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1157990, 31.1022415
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6180573, 30.6066742
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0237885, 33.0138168
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3846207, 33.3497238
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5299225, 41.5140991
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0208588, 44.0076294
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7301331, 27.7255020
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3121414, 26.2937698
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9960091, upper bound: 16.9008482
time: 34.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0648796, upper bound: 16.8749917
time: 41.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.7042503, 19.6845932
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9746933, 19.9486771
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8103943, 24.7954941
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7320862, 21.6920929
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8812714, 24.8747711
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5676460, 22.5342064
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0302353, 30.0055084
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7009583, 25.7128983
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.6964874, 30.7249069
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4295425, 39.4356537
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3954620, 34.3383179
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7651520, 20.7577324
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2882614, 43.2720947
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2080307, 19.2072830
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1747131, 23.1727295
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8171844, 26.8154221
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3575821, 27.3654022
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9656754
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0764008, 37.0933838
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1022339, 31.1208572
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6066742, 30.6210709
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0138092, 33.0273514
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3497238, 33.3923569
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5140991, 41.5383759
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0076294, 44.0285797
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7255020, 27.7340240
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.2937698, 26.3169937
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.8749917, upper bound: 17.0648796
time: 31.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9008482, upper bound: 16.9960091
time: 35.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 69.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 69.93
Output dim: 12, lower bound: -16.9960091, upper bound: 16.9008482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.93
Output dim: 12, lower bound: -17.0648796, upper bound: 16.8749917
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.93
Output dim: 12, lower bound: -16.8749917, upper bound: 17.0648796
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 69.93
Output dim: 12, lower bound: -16.9008482, upper bound: 16.9960091

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6791725, 19.6977654
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9414520, 19.9658051
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.7879868, 24.7998581
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.6755829, 21.7154121
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8793869, 24.8864098
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5260620, 22.5606155
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -29.9910507, 30.0134201
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7088509, 25.6987953
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7128906, 30.6806602
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4231873, 39.4240570
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3628464, 34.4144135
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7514114, 20.7585640
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2735214, 43.2845917
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2097015, 19.2084732
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1734238, 23.1757927
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8173103, 26.8163414
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3669739, 27.3611984
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9692078, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0928116, 37.0803452
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1253510, 31.1113129
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6273346, 30.6134033
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0196075, 33.0077591
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3911667, 33.3568039
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5465088, 41.5272827
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0317078, 44.0166245
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7285080, 27.7247238
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3077164, 26.2893524
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0388722, upper bound: 16.8724666
time: 60.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0622822, upper bound: 16.8489462
time: 34.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6996193, 19.6791763
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9690170, 19.9414558
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8029785, 24.7879868
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7202225, 21.6755829
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8864059, 24.8787613
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5642853, 22.5260620
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0181885, 29.9910507
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.6987953, 25.7109871
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.6806564, 30.7129745
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4240570, 39.4304276
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4236984, 34.3628464
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7603760, 20.7514114
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2894211, 43.2735291
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2084732, 19.2075958
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1757889, 23.1736183
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8189583, 26.8173141
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3611984, 27.3682175
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9692078
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0803452, 37.0970001
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1113129, 31.1304016
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6134033, 30.6303635
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0077515, 33.0231628
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3568039, 33.3989105
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5272827, 41.5549774
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0166321, 44.0394211
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7247314, 27.7324066
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.2893524, 26.3125763
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.8489462, upper bound: 17.0622822
time: 31.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8724666, upper bound: 17.0388722
time: 38.70 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 72.18 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 72.18
Output dim: 12, lower bound: -17.0388722, upper bound: 16.8724666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.18
Output dim: 12, lower bound: -17.0622822, upper bound: 16.8489462
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.18
Output dim: 12, lower bound: -16.8489462, upper bound: 17.0622822
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 72.18
Output dim: 12, lower bound: -16.8724666, upper bound: 17.0388722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6783104, 19.6977654
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9394531, 19.9658051
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.7844391, 24.7998581
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.6740723, 21.7154121
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8785934, 24.8864098
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5243950, 22.5606155
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -29.9846954, 30.0134201
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7088509, 25.6961823
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7126541, 30.6806602
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4231873, 39.4178009
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3628464, 34.4119415
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7486725, 20.7585640
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2734528, 43.2845917
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2097015, 19.2051010
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1734238, 23.1697197
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8173103, 26.8152122
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3669739, 27.3580246
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9655838, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0925369, 37.0803452
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1252441, 31.1113129
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6273346, 30.6097183
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0196075, 33.0069962
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3908997, 33.3568039
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5465088, 41.5270615
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0291443, 44.0166245
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7227173, 27.7247238
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3053589, 26.2893524
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 760

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0617913, upper bound: 16.8343144
time: 29.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0455243, upper bound: 16.8482554
time: 27.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6990471, 19.6783104
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9694824, 19.9394493
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.8034134, 24.7844391
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7189941, 21.6740799
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8859863, 24.8779678
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5634727, 22.5243950
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0190353, 29.9847031
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.6961861, 25.7112007
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.6807785, 30.7127457
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4178009, 39.4309387
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4212341, 34.3600464
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7607193, 20.7486725
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2903137, 43.2734528
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2050934, 19.2061996
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1697159, 23.1744995
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8178291, 26.8176346
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3580170, 27.3686218
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9655800
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0830917, 37.0967407
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1126099, 31.1302948
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6097260, 30.6306992
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0069885, 33.0246887
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3568268, 33.3986511
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5270691, 41.5580673
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0170898, 44.0368576
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7253113, 27.7266159
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.2893753, 26.3102112
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 760

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8482554, upper bound: 17.0455243
time: 36.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.8343144, upper bound: 17.0617913
time: 35.57 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 74.94 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.94
Output dim: 12, lower bound: -17.0617913, upper bound: 16.8343144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.94
Output dim: 12, lower bound: -17.0455243, upper bound: 16.8482554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.94
Output dim: 12, lower bound: -16.8482554, upper bound: 17.0455243
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.94
Output dim: 12, lower bound: -16.8343144, upper bound: 17.0617913

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6724281, 19.6931877
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9382133, 19.9648666
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.7740250, 24.7920303
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.6774216, 21.7190514
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8687744, 24.8760872
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5257988, 22.5623360
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -29.9630127, 29.9970398
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.7102699, 25.6970024
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.7166977, 30.6860733
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4146423, 39.4066086
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.3614044, 34.4122467
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7354164, 20.7485008
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.2837524, 43.2990723
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2107658, 19.2057571
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1596680, 23.1515884
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8171234, 26.8149567
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3587036, 27.3470993
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9656067, 26.9709129
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0967102, 37.0859985
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1260300, 31.1122894
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6222382, 30.6029739
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -33.0126801, 32.9978180
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3955841, 33.3613586
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5528564, 41.5314941
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0090942, 44.0006866
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7119141, 27.7163925
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.3010559, 26.2856941
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0570952, upper bound: 16.8184104
time: 35.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -17.0459515, upper bound: 16.8294747
time: 37.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3742962, 1.6072526, -31.3742962, 1.6072526, -32.9815483, 32.9815483
1: -14.7541466, 4.6208868, -14.7541466, 4.6208868, -19.3750343, 19.3750343
2: -19.7464962, 1.6435061, -19.7464962, 1.6435061, -19.6945381, 19.6724243
3: -19.3851738, 1.1834855, -19.3851738, 1.1834855, -19.9685478, 19.9382133
4: -26.5719547, 0.1655364, -26.5719547, 0.1655364, -24.7955551, 24.7740250
5: -23.2568340, 1.9453506, -23.2568340, 1.9453506, -21.7226334, 21.6774139
6: -18.7368832, 6.9280491, -18.7368832, 6.9280491, -24.8756332, 24.8681488
7: -24.9245796, 0.8847256, -24.9245796, 0.8847256, -22.5651970, 22.5257988
8: -35.0575371, -2.1480403, -35.0575371, -2.1480403, -30.0026398, 29.9630127
9: -13.0921822, 11.7835064, -13.0921822, 11.7835064, -24.8756886, 24.8756886
10: -12.9148941, 18.0476513, -12.9148941, 18.0476513, -30.9625454, 30.9625454
11: -12.3021135, 14.5755329, -12.3021135, 14.5755329, -25.6969948, 25.7126083
12: -0.3073530, 32.9067993, -0.3073530, 32.9067993, -30.6863098, 30.7167854
13: -21.0100574, 19.0525780, -21.0100574, 19.0525780, -39.4066010, 39.4224167
14: -45.0518723, -0.7165356, -45.0518723, -0.7165356, -34.4215317, 34.3585892
15: -22.5688343, -0.4545846, -22.5688343, -0.4545846, -20.7506599, 20.7354202
16: -18.1115780, 7.2155647, -18.1115780, 7.2155647, -25.3271427, 25.3271427
17: -34.6772156, 10.8482285, -34.6772156, 10.8482285, -43.3047943, 43.2837601
18: -11.9880037, 14.3504229, -11.9880037, 14.3504229, -26.3384266, 26.3384266
19: -15.7952881, 1.9880106, -15.7952881, 1.9880106, -17.7832985, 17.7832985
20: -10.9328098, 5.0609274, -10.9328098, 5.0609274, -15.9937372, 15.9937372
21: -12.0918446, 7.5242505, -12.0918446, 7.5242505, -19.6160946, 19.6160946
22: -13.9696035, 7.2737107, -13.9696035, 7.2737107, -21.2433147, 21.2433147
23: -12.6834116, 8.1104689, -12.6834116, 8.1104689, -19.2057533, 19.2072639
24: -18.1477966, 7.0096798, -18.1477966, 7.0096798, -25.1574764, 25.1574764
25: -10.8940535, 8.3322392, -10.8940535, 8.3322392, -19.2262917, 19.2262917
26: -14.2110081, 15.6007442, -14.2110081, 15.6007442, -29.8117523, 29.8117523
27: -24.8063526, 7.0900793, -24.8063526, 7.0900793, -31.8964310, 31.8964310
28: -14.8843174, 9.0277252, -14.8843174, 9.0277252, -23.1515884, 23.1607361
29: -18.2014618, 10.1862745, -18.2014618, 10.1862745, -26.8175735, 26.8174400
30: -14.8017139, 13.3294849, -14.8017139, 13.3294849, -27.3470993, 27.3603439
31: -17.5289440, 2.9215670, -17.5289440, 2.9215670, -20.4505119, 20.4505119
32: -13.3200464, 13.6508675, -13.3200464, 13.6508675, -26.9709129, 26.9656067
33: -32.1030960, 6.8771768, -32.1030960, 6.8771768, -37.0887146, 37.1009216
34: -25.7192039, 7.8386326, -25.7192039, 7.8386326, -31.1135788, 31.1310730
35: -19.9621334, 11.4848595, -19.9621334, 11.4848595, -30.6029816, 30.6256104
36: -19.9939632, 15.7010365, -19.9939632, 15.7010365, -32.9978180, 33.0178146
37: -28.5910053, 7.4333568, -28.5910053, 7.4333568, -33.3614044, 33.4033127
38: -25.0449867, 16.8481216, -25.0449867, 16.8481216, -41.5314941, 41.5644073
39: -36.4644699, 8.6757545, -36.4644699, 8.6757545, -44.0012054, 44.0168152
40: -31.1685467, 1.0877852, -31.1685467, 1.0877852, -27.7169800, 27.7158279
41: -18.2469711, 11.9246311, -18.2469711, 11.9246311, -26.2857513, 26.3059120
42: -11.3728771, 11.5306053, -11.3728771, 11.5306053, -22.9034824, 22.9034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8294747, upper bound: 17.0459515
time: 33.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8184104, upper bound: 17.0570952
time: 35.00 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 71.47 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 71.47
Output dim: 12, lower bound: -17.0570952, upper bound: 16.8184104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 71.47
Output dim: 12, lower bound: -17.0459515, upper bound: 16.8294747
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 71.47
Output dim: 12, lower bound: -16.8294747, upper bound: 17.0459515
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 71.47
Output dim: 12, lower bound: -16.8184104, upper bound: 17.0570952

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 93.55 + 1092.12 = 1185.67 seconds
