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
execution time: IAR + RelationalAnalysis = 3.02 + 91.79 = 94.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -17.0952923, upper bound: 17.0952923

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1311
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1658

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0315090
time: 27.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0861274
time: 33.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 61.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 61.27
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0315090
IS_A2, status: Status.UNKNOWN, split count: 1, time: 61.27
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0861274

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -31.3369331, 1.5973544, -31.3619938, 1.6040959, -32.9410286, 32.9593468
1: -14.7415524, 4.6152139, -14.7500229, 4.6190462, -19.3605995, 19.3652363
2: -19.7089157, 1.6343544, -19.7342110, 1.6405451, -19.6734810, 19.6945229
3: -19.3697472, 1.1761107, -19.3801689, 1.1810818, -19.9726105, 19.9759560
4: -26.5217571, 0.1574016, -26.5556641, 0.1629167, -24.7628403, 24.7936478
5: -23.2321701, 1.9354823, -23.2486095, 1.9421847, -21.7363167, 21.7442055
6: -18.7235508, 6.9033952, -18.7325401, 6.9199491, -24.8476868, 24.8321609
7: -24.9058914, 0.8760743, -24.9183922, 0.8819332, -22.5713310, 22.5761909
8: -35.0015869, -2.1598496, -35.0393219, -2.1518526, -29.9827499, 30.0166168
9: -13.0824089, 11.7663746, -13.0890102, 11.7779589, -24.8603668, 24.8553848
10: -12.9014530, 18.0327530, -12.9104395, 18.0428543, -30.9443073, 30.9431915
11: -12.2897377, 14.5553303, -12.2981377, 14.5688963, -25.6921387, 25.6844864
12: -0.2915449, 32.8434830, -0.3022032, 32.8862953, -30.7131577, 30.6789474
13: -20.9985809, 19.0312290, -21.0063057, 19.0454388, -39.4204407, 39.4128189
14: -45.0377846, -0.7379839, -45.0472832, -0.7235155, -34.3680954, 34.3538742
15: -22.5308800, -0.4597449, -22.5560799, -0.4562645, -20.7275925, 20.7535820
16: -18.0983009, 7.2002354, -18.1072826, 7.2104855, -25.3087864, 25.3075180
17: -34.6599884, 10.7672863, -34.6716614, 10.8220634, -43.2532349, 43.2103043
18: -11.9690018, 14.3399010, -11.9818478, 14.3469915, -26.3159943, 26.3217487
19: -15.7814560, 1.9833186, -15.7907829, 1.9864721, -17.7679291, 17.7741013
20: -10.9102945, 5.0548396, -10.9253941, 5.0589728, -15.9692669, 15.9802341
21: -12.0766258, 7.5180626, -12.0869198, 7.5222197, -19.5988464, 19.6049824
22: -13.9531384, 7.2448339, -13.9642839, 7.2643824, -21.2175217, 21.2091179
23: -12.6731663, 8.1071882, -12.6800728, 8.1093922, -19.1981201, 19.2033844
24: -18.1231270, 6.9998035, -18.1398582, 7.0064421, -25.1295700, 25.1396618
25: -10.8727665, 8.3262072, -10.8871613, 8.3302689, -19.2030354, 19.2133675
26: -14.1938992, 15.5878086, -14.2054663, 15.5963802, -29.7902794, 29.7932739
27: -24.7841320, 7.0713406, -24.7991676, 7.0838728, -31.8680038, 31.8705082
28: -14.8734550, 8.9989843, -14.8808298, 9.0181084, -23.1487427, 23.1323929
29: -18.1848068, 10.1309376, -18.1960793, 10.1683722, -26.7798996, 26.7528152
30: -14.7925987, 13.2933331, -14.7987757, 13.3177071, -27.3416138, 27.3169785
31: -17.5146885, 2.9166169, -17.5243092, 2.9199386, -20.4346275, 20.4409256
32: -13.3062572, 13.6143236, -13.3155546, 13.6389561, -26.9401169, 26.9177895
33: -32.0435257, 6.8658075, -32.0836792, 6.8734937, -37.0304718, 37.0634308
34: -25.6890297, 7.8314409, -25.7092896, 7.8363233, -31.0819702, 31.0964890
35: -19.9413948, 11.4816475, -19.9553413, 11.4838123, -30.5814514, 30.5989304
36: -19.9767876, 15.6947241, -19.9884014, 15.6989784, -33.0161896, 33.0216827
37: -28.5326443, 7.4289432, -28.5720463, 7.4319177, -33.3420105, 33.3778152
38: -24.9676876, 16.8412971, -25.0199966, 16.8458824, -41.4268036, 41.4791565
39: -36.4003754, 8.6689034, -36.4436340, 8.6735077, -43.9420319, 43.9855270
40: -31.1339397, 1.0825939, -31.1572094, 1.0860896, -27.7216263, 27.7381287
41: -18.2262325, 11.9124346, -18.2402267, 11.9206333, -26.2999878, 26.3041687
42: -11.3563662, 11.5237198, -11.3674850, 11.5283527, -22.8847198, 22.8912048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1311
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1399

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
time: 29.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
time: 35.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -31.3993874, 1.6790943, -31.3720360, 1.6068134, -33.0062027, 33.0511322
1: -14.7799816, 4.6784000, -14.7528620, 4.6205912, -19.4005737, 19.4312630
2: -19.7516289, 1.6946530, -19.7445259, 1.6431451, -19.7213249, 19.7664948
3: -19.3974590, 1.2260098, -19.3835144, 1.1828704, -20.0046997, 20.0288010
4: -26.5803947, 0.2446356, -26.5692635, 0.1651936, -24.8265762, 24.8996582
5: -23.2651863, 2.0044403, -23.2541313, 1.9443676, -21.7680779, 21.8193626
6: -18.7927399, 6.9388561, -18.7364578, 6.9257956, -24.9458008, 24.8705826
7: -24.9496536, 0.9519176, -24.9225311, 0.8844233, -22.6219254, 22.6626816
8: -35.0687485, -2.0633192, -35.0543289, -2.1485295, -30.0554047, 30.1324310
9: -13.1111298, 11.8240566, -13.0914993, 11.7809095, -24.8920403, 24.9155560
10: -12.9368458, 18.1182995, -12.9133244, 18.0468082, -30.9836540, 31.0316238
11: -12.4172173, 14.5860882, -12.3015604, 14.5740309, -25.8328476, 25.7228012
12: -0.4431181, 32.9190292, -0.3064709, 32.9045715, -30.8864517, 30.7536583
13: -21.0455494, 19.0804749, -21.0093193, 19.0513058, -39.4731369, 39.4665680
14: -45.1098480, -0.7032046, -45.0506821, -0.7186165, -34.4730759, 34.3947906
15: -22.5904675, -0.3631158, -22.5655136, -0.4549208, -20.7953224, 20.8630524
16: -18.1698246, 7.2623801, -18.1103210, 7.2139807, -25.3838043, 25.3727016
17: -34.9269257, 10.8663979, -34.6763000, 10.8457823, -43.5466614, 43.3101425
18: -12.0173216, 14.3797855, -11.9862881, 14.3498030, -26.3671246, 26.3660736
19: -15.8380146, 2.0030890, -15.7947178, 1.9878561, -17.8258705, 17.7978058
20: -10.9628181, 5.1012721, -10.9313831, 5.0606685, -16.0234871, 16.0326557
21: -12.1674080, 7.5451784, -12.0910769, 7.5238752, -19.6912842, 19.6362553
22: -14.0862389, 7.2967138, -13.9688730, 7.2726183, -21.3588562, 21.2655869
23: -12.7146435, 8.1270561, -12.6827106, 8.1100779, -19.2460938, 19.2264748
24: -18.1745853, 7.0443206, -18.1443844, 7.0090642, -25.1836491, 25.1887054
25: -10.9245300, 8.3698463, -10.8907843, 8.3318462, -19.2563763, 19.2606316
26: -14.2991047, 15.6028347, -14.2102194, 15.5992165, -29.8983212, 29.8130531
27: -24.8442287, 7.0934901, -24.8051300, 7.0876827, -31.9319115, 31.8986206
28: -14.9390240, 9.0323267, -14.8836908, 9.0257206, -23.2338867, 23.1712990
29: -18.3544197, 10.1921158, -18.2009201, 10.1845436, -26.9691010, 26.8194199
30: -14.8670168, 13.3501682, -14.8010998, 13.3279781, -27.4431915, 27.3729553
31: -17.5584164, 2.9541407, -17.5278244, 2.9213343, -20.4797516, 20.4819641
32: -13.3840351, 13.6604652, -13.3193645, 13.6470108, -27.0310459, 26.9734116
33: -32.1237679, 6.9733143, -32.0997620, 6.8768177, -37.1154785, 37.1943054
34: -25.7394218, 7.8856831, -25.7176876, 7.8383288, -31.1367645, 31.1642303
35: -19.9929314, 11.5323410, -19.9606915, 11.4847660, -30.6317596, 30.6671448
36: -20.0496407, 15.7144232, -19.9932766, 15.7000933, -33.0950165, 33.0502777
37: -28.6376171, 7.5078835, -28.5879383, 7.4331226, -33.4454498, 33.4756851
38: -25.0868416, 16.9448681, -25.0426559, 16.8470535, -41.5417023, 41.6178818
39: -36.4803925, 8.7731171, -36.4618835, 8.6753778, -44.0220184, 44.1243591
40: -31.1952248, 1.1601367, -31.1665020, 1.0874529, -27.7790070, 27.8310013
41: -18.2761784, 11.9284163, -18.2456627, 11.9222984, -26.3599319, 26.3296280
42: -11.4112206, 11.5472794, -11.3723440, 11.5299311, -22.9411507, 22.9196243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1399

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
time: 31.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
time: 34.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 69.22 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 69.22
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 69.22
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 69.22
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 69.22
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -31.3922520, 1.6764588, -31.3524132, 1.6052065, -32.9974594, 33.0288734
1: -14.7781420, 4.6764765, -14.7488909, 4.6177311, -19.3958740, 19.4253674
2: -19.7360287, 1.6924493, -19.6973991, 1.6416135, -19.7036743, 19.7179108
3: -19.3900070, 1.2237346, -19.3631611, 1.1815791, -20.0060272, 20.0045776
4: -26.5530052, 0.2418489, -26.4889107, 0.1683283, -24.8004303, 24.8158722
5: -23.2487564, 2.0019507, -23.2052479, 1.9457865, -21.7576180, 21.7860298
6: -18.7838821, 6.9340148, -18.7115440, 6.9135871, -24.9218597, 24.8403702
7: -24.9309902, 0.9494991, -24.8677502, 0.8838644, -22.5907707, 22.6025620
8: -35.0560226, -2.0677638, -35.0189209, -2.1574864, -30.0352783, 30.0925980
9: -13.1088047, 11.8177242, -13.0864506, 11.7637148, -24.8725204, 24.9041748
10: -12.9323368, 18.0944653, -12.9126930, 17.9762878, -30.9086246, 31.0071583
11: -12.4144821, 14.5757999, -12.3044310, 14.5446119, -25.8013191, 25.7154045
12: -0.4375696, 32.8823738, -0.3000822, 32.7928925, -30.7658539, 30.6999969
13: -21.0386467, 19.0778122, -20.9928112, 19.0505104, -39.4658051, 39.4468994
14: -45.1055260, -0.7258277, -45.0509949, -0.7852001, -34.3907471, 34.3513489
15: -22.5847301, -0.3649907, -22.5487499, -0.4522352, -20.7876282, 20.8467178
16: -18.1665306, 7.2585974, -18.1072865, 7.2058954, -25.3724251, 25.3658829
17: -34.9221725, 10.8364830, -34.6824493, 10.7558470, -43.4483490, 43.2843399
18: -12.0097446, 14.3756351, -11.9694881, 14.3410559, -26.3507996, 26.3451233
19: -15.8271484, 2.0012553, -15.7642899, 1.9867358, -17.8138847, 17.7655449
20: -10.9491587, 5.0992947, -10.8921242, 5.0573368, -16.0064964, 15.9914188
21: -12.1631174, 7.5437040, -12.0833645, 7.5255990, -19.6887169, 19.6270676
22: -14.0810308, 7.2940254, -13.9596148, 7.2654128, -21.3464432, 21.2536392
23: -12.7123432, 8.1233006, -12.6806793, 8.1002722, -19.2343292, 19.2212753
24: -18.1704330, 7.0412045, -18.1348495, 7.0037627, -25.1741962, 25.1760540
25: -10.9216862, 8.3652630, -10.8860912, 8.3198442, -19.2415314, 19.2513542
26: -14.2938271, 15.5932102, -14.2065554, 15.5706406, -29.8644676, 29.7997665
27: -24.8255692, 7.0898676, -24.7540092, 7.0760112, -31.9015808, 31.8438759
28: -14.9368706, 9.0306721, -14.8833389, 9.0234108, -23.2278824, 23.1688309
29: -18.3499832, 10.1759157, -18.1992569, 10.1356392, -26.9150887, 26.8001671
30: -14.8645868, 13.3372526, -14.8019419, 13.2907524, -27.4043808, 27.3606796
31: -17.5419216, 2.9517460, -17.4799385, 2.9208102, -20.4627323, 20.4316845
32: -13.3743877, 13.6574535, -13.2927227, 13.6419115, -27.0163002, 26.9410553
33: -32.1060638, 6.9707661, -32.0488586, 6.8800378, -37.0993500, 37.1371384
34: -25.7339058, 7.8824120, -25.7100735, 7.8315730, -31.1206512, 31.1474838
35: -19.9827995, 11.5308800, -19.9324799, 11.4872284, -30.6237640, 30.6384888
36: -20.0327644, 15.7133789, -19.9439945, 15.7051716, -33.0826187, 32.9994507
37: -28.6203709, 7.5057960, -28.5404816, 7.4348707, -33.4306183, 33.4236755
38: -25.0640087, 16.9425335, -24.9780350, 16.8535862, -41.5209656, 41.5460663
39: -36.4493942, 8.7709751, -36.3712959, 8.6847639, -43.9998474, 44.0304108
40: -31.1748924, 1.1577868, -31.1068268, 1.0913982, -27.7507172, 27.7572289
41: -18.2631989, 11.9259443, -18.2091484, 11.9116573, -26.3407974, 26.2933426
42: -11.4075470, 11.5316372, -11.3694210, 11.4853001, -22.8928471, 22.9010582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
time: 36.32 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434
time: 34.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -31.3993874, 1.6790943, -31.3682785, 1.6044755, -33.0038643, 33.0473709
1: -14.7799816, 4.6784000, -14.7516212, 4.6192279, -19.3992100, 19.4300213
2: -19.7516289, 1.6946530, -19.7396984, 1.6417475, -19.7195358, 19.7514191
3: -19.3974590, 1.2260098, -19.3807602, 1.1804240, -20.0011482, 20.0445099
4: -26.5803947, 0.2446356, -26.5588779, 0.1634445, -24.8246536, 24.8823700
5: -23.2651863, 2.0044403, -23.2487411, 1.9418206, -21.7656822, 21.8063850
6: -18.7927399, 6.9388561, -18.7334251, 6.9229565, -24.9389191, 24.8690491
7: -24.9496536, 0.9519176, -24.9165382, 0.8824444, -22.6196976, 22.6222496
8: -35.0687485, -2.0633192, -35.0488052, -2.1505318, -30.0531921, 30.1223145
9: -13.1111298, 11.8240566, -13.0904016, 11.7772102, -24.8883400, 24.9144592
10: -12.9368458, 18.1182995, -12.9110670, 18.0380135, -30.9748592, 31.0293655
11: -12.4172173, 14.5860882, -12.2995138, 14.5702744, -25.8203125, 25.7206535
12: -0.4431181, 32.9190292, -0.3030758, 32.8934860, -30.8320084, 30.7497177
13: -21.0455494, 19.0804749, -21.0065613, 19.0492706, -39.4709167, 39.4642715
14: -45.1098480, -0.7032046, -45.0471497, -0.7264042, -34.4160004, 34.3909988
15: -22.5904675, -0.3631158, -22.5627670, -0.4562550, -20.7932663, 20.8603363
16: -18.1698246, 7.2623801, -18.1084042, 7.2110462, -25.3808708, 25.3707848
17: -34.9269257, 10.8663979, -34.6721153, 10.8350964, -43.5171051, 43.3057251
18: -12.0173216, 14.3797855, -11.9814482, 14.3478546, -26.3651772, 26.3612328
19: -15.8380146, 2.0030890, -15.7899675, 1.9868848, -17.8248997, 17.7930565
20: -10.9628181, 5.1012721, -10.9267826, 5.0594926, -16.0223103, 16.0280552
21: -12.1674080, 7.5451784, -12.0883751, 7.5231185, -19.6905270, 19.6335526
22: -14.0862389, 7.2967138, -13.9651937, 7.2715616, -21.3577995, 21.2619076
23: -12.7146435, 8.1270561, -12.6812935, 8.1082096, -19.2427979, 19.2248955
24: -18.1745853, 7.0443206, -18.1409988, 7.0075569, -25.1821423, 25.1853199
25: -10.9245300, 8.3698463, -10.8883934, 8.3297148, -19.2542458, 19.2582397
26: -14.2991047, 15.6028347, -14.2056160, 15.5955820, -29.8946877, 29.8084507
27: -24.8442287, 7.0934901, -24.8004951, 7.0862150, -31.9304428, 31.8939857
28: -14.9390240, 9.0323267, -14.8822088, 9.0248232, -23.2328110, 23.1693420
29: -18.3544197, 10.1921158, -18.1978149, 10.1785898, -26.9470139, 26.8162880
30: -14.8670168, 13.3501682, -14.7992973, 13.3228722, -27.4342194, 27.3710632
31: -17.5584164, 2.9541407, -17.5213928, 2.9201593, -20.4785767, 20.4755325
32: -13.3840351, 13.6604652, -13.3151531, 13.6450844, -27.0291195, 26.9673271
33: -32.1237679, 6.9733143, -32.0929565, 6.8751612, -37.1135101, 37.1781769
34: -25.7394218, 7.8856831, -25.7154140, 7.8367271, -31.1336212, 31.1641846
35: -19.9929314, 11.5323410, -19.9566841, 11.4835453, -30.6308365, 30.6631317
36: -20.0496407, 15.7144232, -19.9868202, 15.6994667, -33.0941620, 33.0358353
37: -28.6376171, 7.5078835, -28.5814819, 7.4312129, -33.4435425, 33.4650726
38: -25.0868416, 16.9448681, -25.0334244, 16.8455353, -41.5425873, 41.6077118
39: -36.4803925, 8.7731171, -36.4501266, 8.6738491, -44.0202484, 44.1080170
40: -31.1952248, 1.1601367, -31.1598892, 1.0854383, -27.7769623, 27.7793732
41: -18.2761784, 11.9284163, -18.2424622, 11.9204254, -26.3579788, 26.3149414
42: -11.4112206, 11.5472794, -11.3703976, 11.5219593, -22.9331799, 22.9176769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1473
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1311
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0114178, upper bound: 17.0707434
time: 33.82 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
time: 30.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 66.62 seconds
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 66.62
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 66.62
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 66.62
Output dim: 12, lower bound: -17.0114178, upper bound: 17.0707434
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 66.62
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -31.3810043, 1.6735220, -31.3063831, 1.5499930, -32.9309959, 32.9799042
1: -14.7751627, 4.6747923, -14.7247362, 4.5751863, -19.3503494, 19.3995285
2: -19.7265072, 1.6901679, -19.6646729, 1.6063371, -19.6566124, 19.6806221
3: -19.3839607, 1.2215104, -19.3368015, 1.1397173, -19.9586563, 19.9729576
4: -26.5382538, 0.2397623, -26.4382305, 0.1170940, -24.7338867, 24.7628860
5: -23.2412071, 1.9988818, -23.1758537, 1.8950927, -21.6993828, 21.7508392
6: -18.7809334, 6.9215040, -18.6591873, 6.8672924, -24.8718414, 24.7682571
7: -24.9257011, 0.9472613, -24.8354874, 0.8365741, -22.5341415, 22.5648117
8: -35.0370216, -2.0711780, -34.9538765, -2.2219934, -29.9497528, 30.0222626
9: -13.1059151, 11.8140726, -13.0649519, 11.7244987, -24.8304138, 24.8790245
10: -12.9277601, 18.0905685, -12.8811007, 17.9227676, -30.8505287, 30.9716682
11: -12.4112301, 14.5718145, -12.2371559, 14.5239573, -25.7757568, 25.6391563
12: -0.4329586, 32.8624344, -0.1920500, 32.7243805, -30.6910095, 30.5706749
13: -21.0350666, 19.0685539, -20.9505997, 19.0053959, -39.4100189, 39.3889999
14: -45.1013222, -0.7305763, -45.0064964, -0.8107862, -34.3565445, 34.2773666
15: -22.5720558, -0.3664393, -22.4928322, -0.5160561, -20.7109985, 20.7893066
16: -18.1635742, 7.2543116, -18.0717583, 7.1617908, -25.3253651, 25.3260689
17: -34.9177475, 10.8210831, -34.5384750, 10.7009754, -43.3883362, 43.1244354
18: -12.0047131, 14.3728428, -11.9347878, 14.3123646, -26.3170776, 26.3076305
19: -15.8234253, 2.0002527, -15.7274218, 1.9749012, -17.7983265, 17.7276745
20: -10.9441032, 5.0977144, -10.8557777, 5.0329752, -15.9770784, 15.9534922
21: -12.1588802, 7.5424061, -12.0259838, 7.5079293, -19.6668091, 19.5683899
22: -14.0761795, 7.2881684, -13.8841095, 7.2336993, -21.3098793, 21.1722775
23: -12.7084827, 8.1218710, -12.6479282, 8.0802536, -19.2092514, 19.1842384
24: -18.1623402, 7.0386772, -18.0953064, 6.9669824, -25.1293221, 25.1339836
25: -10.9161472, 8.3635101, -10.8452682, 8.2914152, -19.2075615, 19.2087784
26: -14.2890005, 15.5919571, -14.1362314, 15.5624962, -29.8514977, 29.7281876
27: -24.8197384, 7.0867577, -24.7210236, 7.0581036, -31.8778419, 31.8077812
28: -14.9340706, 9.0258904, -14.8466043, 9.0028429, -23.2036438, 23.1231499
29: -18.3456078, 10.1641827, -18.1121159, 10.0945606, -26.8685150, 26.6981735
30: -14.8622141, 13.3331108, -14.7695131, 13.2643461, -27.3729401, 27.3145218
31: -17.5379009, 2.9502544, -17.4470291, 2.8955054, -20.4334068, 20.3972836
32: -13.3710537, 13.6435242, -13.2338734, 13.5897102, -26.9607639, 26.8633041
33: -32.0925980, 6.9680004, -31.9889088, 6.8272505, -37.0262375, 37.0698853
34: -25.7241020, 7.8801126, -25.6671276, 7.7918377, -31.0609436, 31.0972061
35: -19.9769783, 11.5299873, -19.8948288, 11.4578028, -30.5790176, 30.5963440
36: -20.0282707, 15.7092037, -19.8927803, 15.6829348, -33.0521698, 32.9404602
37: -28.6048050, 7.5044746, -28.4654503, 7.3931022, -33.3690643, 33.3437271
38: -25.0513954, 16.9406967, -24.9105492, 16.8126812, -41.4548035, 41.4761887
39: -36.4349823, 8.7695246, -36.3146210, 8.6453514, -43.9333954, 43.9671249
40: -31.1651630, 1.1557302, -31.0638027, 1.0505166, -27.6943207, 27.7118111
41: -18.2585411, 11.9218588, -18.1794472, 11.8925810, -26.3139572, 26.2517700
42: -11.4036579, 11.5290833, -11.3320122, 11.4690247, -22.8726826, 22.8610954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=193, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1416

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9164283, upper bound: 16.9401857
time: 35.15 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9165777, upper bound: 17.0582654
time: 31.96 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -31.3908081, 1.6762099, -31.3479767, 1.6044164, -32.9952240, 33.0241852
1: -14.7773571, 4.6763124, -14.7465677, 4.6172915, -19.3946495, 19.4228802
2: -19.7348824, 1.6922522, -19.6938667, 1.6410022, -19.7018509, 19.7099075
3: -19.3891869, 1.2233593, -19.3608475, 1.1803627, -20.0036201, 19.9927597
4: -26.5516586, 0.2416558, -26.4847088, 0.1677837, -24.7985229, 24.8056946
5: -23.2471504, 2.0013468, -23.2002754, 1.9439120, -21.7535515, 21.7660370
6: -18.7836170, 6.9329176, -18.7107964, 6.9101820, -24.9163055, 24.8440361
7: -24.9296799, 0.9492965, -24.8640079, 0.8833156, -22.5883331, 22.5869751
8: -35.0544128, -2.0680585, -35.0138092, -2.1583676, -30.0327225, 30.0751114
9: -13.1083965, 11.8170090, -13.0851631, 11.7612743, -24.8696709, 24.9021721
10: -12.9311438, 18.0939960, -12.9090405, 17.9749107, -30.9060555, 31.0030365
11: -12.4142418, 14.5752306, -12.3036585, 14.5428505, -25.7980499, 25.7141113
12: -0.4370394, 32.8807220, -0.2984200, 32.7877502, -30.7443390, 30.6967316
13: -21.0381508, 19.0765343, -20.9911976, 19.0463886, -39.4560852, 39.4438705
14: -45.1050339, -0.7264016, -45.0495529, -0.7872140, -34.3822937, 34.3615875
15: -22.5832920, -0.3651342, -22.5443916, -0.4526677, -20.7854691, 20.8358040
16: -18.1659966, 7.2581768, -18.1056557, 7.2046413, -25.3706379, 25.3638325
17: -34.9218140, 10.8351154, -34.6813889, 10.7517204, -43.4412994, 43.2817078
18: -12.0087299, 14.3752584, -11.9665966, 14.3399086, -26.3486385, 26.3418541
19: -15.8268356, 2.0011675, -15.7632980, 1.9864311, -17.8132668, 17.7644653
20: -10.9484997, 5.0991130, -10.8902454, 5.0568051, -16.0053043, 15.9893589
21: -12.1627388, 7.5435791, -12.0822382, 7.5251021, -19.6878414, 19.6258163
22: -14.0805798, 7.2933908, -13.9582767, 7.2634101, -21.3439903, 21.2516670
23: -12.7114277, 8.1230602, -12.6777229, 8.0995073, -19.2327232, 19.2177391
24: -18.1696892, 7.0408664, -18.1328278, 7.0027037, -25.1723938, 25.1736946
25: -10.9200535, 8.3650370, -10.8809719, 8.3191032, -19.2391567, 19.2460098
26: -14.2934065, 15.5929756, -14.2053509, 15.5699730, -29.8633804, 29.7983265
27: -24.8247375, 7.0894480, -24.7516899, 7.0748391, -31.8995762, 31.8411369
28: -14.9365768, 9.0298405, -14.8823938, 9.0207644, -23.2242661, 23.1686096
29: -18.3496170, 10.1750374, -18.1981888, 10.1328955, -26.9111557, 26.7984924
30: -14.8643990, 13.3365974, -14.8013296, 13.2885380, -27.3995514, 27.3613358
31: -17.5415916, 2.9515748, -17.4788857, 2.9202933, -20.4618855, 20.4304600
32: -13.3740082, 13.6562920, -13.2915134, 13.6383896, -27.0123978, 26.9419212
33: -32.1041107, 6.9705186, -32.0429306, 6.8793602, -37.0966797, 37.1295547
34: -25.7329617, 7.8821564, -25.7071304, 7.8308620, -31.1219025, 31.1413345
35: -19.9820538, 11.5308208, -19.9302673, 11.4869843, -30.6276093, 30.6325073
36: -20.0323410, 15.7121048, -19.9425907, 15.7014532, -33.0729523, 32.9966736
37: -28.6183319, 7.5056267, -28.5344028, 7.4344501, -33.4291534, 33.4166031
38: -25.0626297, 16.9420547, -24.9738674, 16.8521385, -41.5286407, 41.5389862
39: -36.4479599, 8.7708359, -36.3668938, 8.6843901, -44.0034485, 44.0216293
40: -31.1731853, 1.1575785, -31.1018810, 1.0907044, -27.7476120, 27.7442474
41: -18.2626915, 11.9244280, -18.2076721, 11.9069843, -26.3327179, 26.2906380
42: -11.4072437, 11.5313511, -11.3684540, 11.4843712, -22.8916149, 22.8998051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1416

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9164283, upper bound: 16.9401857
time: 37.68 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9759248, upper bound: 17.0582654
time: 35.57 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -31.3881302, 1.6761246, -31.3221931, 1.5492415, -32.9373703, 32.9983177
1: -14.7770281, 4.6766930, -14.7274113, 4.5766721, -19.3537006, 19.4041042
2: -19.7421322, 1.6923556, -19.7069435, 1.6064434, -19.6724548, 19.7141380
3: -19.3914127, 1.2237704, -19.3543777, 1.1385970, -19.9537582, 20.0128670
4: -26.5656528, 0.2425838, -26.5081902, 0.1122026, -24.7580872, 24.8294144
5: -23.2576790, 2.0013399, -23.2193375, 1.8910816, -21.7072563, 21.7711830
6: -18.7897987, 6.9263248, -18.6810684, 6.8766861, -24.8888397, 24.7969742
7: -24.9443741, 0.9496913, -24.8842487, 0.8351531, -22.5630264, 22.5845337
8: -35.0497360, -2.0667253, -34.9837837, -2.2150288, -29.9676895, 30.0520172
9: -13.1082363, 11.8204184, -13.0688791, 11.7379799, -24.8462162, 24.8892975
10: -12.9322929, 18.1144142, -12.8794556, 17.9844971, -30.9167900, 30.9938698
11: -12.4139271, 14.5820971, -12.2322369, 14.5496626, -25.7947235, 25.6444359
12: -0.4384613, 32.8990593, -0.1950045, 32.8249817, -30.7571564, 30.6204109
13: -21.0419464, 19.0711956, -20.9643898, 19.0040455, -39.4151688, 39.4063339
14: -45.1056557, -0.7079883, -45.0025520, -0.7519794, -34.3817749, 34.3170242
15: -22.5777550, -0.3645744, -22.5067139, -0.5200553, -20.7166405, 20.8028946
16: -18.1668720, 7.2580843, -18.0728951, 7.1669054, -25.3337784, 25.3309784
17: -34.9224930, 10.8510017, -34.5281143, 10.7802382, -43.4571381, 43.1457977
18: -12.0122910, 14.3769894, -11.9468002, 14.3191309, -26.3314209, 26.3237896
19: -15.8342667, 2.0020826, -15.7531710, 1.9750354, -17.8093014, 17.7552528
20: -10.9577751, 5.0996909, -10.8904381, 5.0351295, -15.9929047, 15.9901295
21: -12.1631746, 7.5439105, -12.0310078, 7.5054622, -19.6686363, 19.5749187
22: -14.0814028, 7.2908664, -13.8896971, 7.2398643, -21.3212662, 21.1805630
23: -12.7107782, 8.1256056, -12.6484976, 8.0882044, -19.2176933, 19.1878204
24: -18.1664925, 7.0417981, -18.1014652, 6.9707370, -25.1372299, 25.1432629
25: -10.9189835, 8.3680983, -10.8475895, 8.3013153, -19.2202988, 19.2156868
26: -14.2942677, 15.6016054, -14.1353283, 15.5874500, -29.8817177, 29.7369347
27: -24.8383827, 7.0903993, -24.7675190, 7.0683236, -31.9067059, 31.8579178
28: -14.9362068, 9.0275230, -14.8454294, 9.0042477, -23.2085876, 23.1236877
29: -18.3500290, 10.1803904, -18.1106968, 10.1375103, -26.9004669, 26.7142792
30: -14.8646374, 13.3460817, -14.7668705, 13.2964363, -27.4028168, 27.3248901
31: -17.5543900, 2.9526625, -17.4885330, 2.8948579, -20.4492474, 20.4411964
32: -13.3806858, 13.6465225, -13.2563057, 13.5928345, -26.9735203, 26.8895187
33: -32.1103134, 6.9705153, -32.0330200, 6.8223600, -37.0404282, 37.1109161
34: -25.7296143, 7.8834023, -25.6724834, 7.7970419, -31.0738831, 31.1138611
35: -19.9871044, 11.5314236, -19.9190178, 11.4541798, -30.5861359, 30.6209793
36: -20.0451126, 15.7102156, -19.9356613, 15.6772614, -33.0637054, 32.9769211
37: -28.6220341, 7.5065598, -28.5064850, 7.3894634, -33.3819885, 33.3851624
38: -25.0741787, 16.9430485, -24.9659996, 16.8046227, -41.4764557, 41.5378723
39: -36.4660110, 8.7716541, -36.3934822, 8.6343899, -43.9538116, 44.0447159
40: -31.1854897, 1.1580653, -31.1168594, 1.0445700, -27.7205811, 27.7339478
41: -18.2714996, 11.9243612, -18.2127476, 11.9013882, -26.3311157, 26.2733994
42: -11.4073067, 11.5446873, -11.3329992, 11.5056725, -22.9129791, 22.8776855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=193, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1473
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1311
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1399

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9290624, upper bound: 16.9884045
time: 35.01 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
time: 34.85 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -31.3979912, 1.6788025, -31.3638229, 1.6037188, -33.0017090, 33.0426254
1: -14.7792177, 4.6782236, -14.7492962, 4.6187658, -19.3979836, 19.4275208
2: -19.7505112, 1.6944633, -19.7361889, 1.6411383, -19.7177162, 19.7434273
3: -19.3966713, 1.2256262, -19.3784313, 1.1792202, -19.9987335, 20.0326691
4: -26.5790367, 0.2445016, -26.5546532, 0.1629047, -24.8227539, 24.8722000
5: -23.2635918, 2.0038259, -23.2437630, 1.9398873, -21.7615891, 21.7864075
6: -18.7924805, 6.9377527, -18.7326794, 6.9195580, -24.9333496, 24.8727493
7: -24.9483376, 0.9517303, -24.9128170, 0.8818817, -22.6172485, 22.6066742
8: -35.0671425, -2.0636067, -35.0436668, -2.1513877, -30.0506821, 30.1048660
9: -13.1107321, 11.8233299, -13.0891132, 11.7747555, -24.8854866, 24.9124432
10: -12.9356623, 18.1178703, -12.9073906, 18.0366554, -30.9723167, 31.0252609
11: -12.4169617, 14.5855198, -12.2987423, 14.5685129, -25.8170547, 25.7193947
12: -0.4425659, 32.9173622, -0.3014002, 32.8883667, -30.8104553, 30.7464447
13: -21.0450382, 19.0791168, -21.0050125, 19.0451317, -39.4613037, 39.4612274
14: -45.1093864, -0.7037778, -45.0456772, -0.7284327, -34.4075775, 34.4012299
15: -22.5890217, -0.3632598, -22.5583916, -0.4566584, -20.7910995, 20.8494415
16: -18.1692848, 7.2619534, -18.1067619, 7.2097988, -25.3790836, 25.3687153
17: -34.9265633, 10.8650522, -34.6711006, 10.8310232, -43.5101395, 43.3031769
18: -12.0163212, 14.3794165, -11.9785662, 14.3466911, -26.3630123, 26.3579826
19: -15.8376713, 2.0029843, -15.7889748, 1.9865811, -17.8242531, 17.7919598
20: -10.9621592, 5.1011038, -10.9248991, 5.0589523, -16.0211105, 16.0260029
21: -12.1670399, 7.5450478, -12.0872602, 7.5226107, -19.6896515, 19.6323090
22: -14.0857887, 7.2960978, -13.9638577, 7.2695880, -21.3553772, 21.2599564
23: -12.7137299, 8.1268177, -12.6783361, 8.1074667, -19.2411957, 19.2213440
24: -18.1738815, 7.0439863, -18.1389904, 7.0064635, -25.1803455, 25.1829758
25: -10.9228840, 8.3696089, -10.8832703, 8.3289938, -19.2518768, 19.2528801
26: -14.2986832, 15.6026134, -14.2044067, 15.5949326, -29.8936157, 29.8070202
27: -24.8434238, 7.0930862, -24.7981758, 7.0850410, -31.9284649, 31.8912621
28: -14.9387283, 9.0314732, -14.8812857, 9.0221558, -23.2292099, 23.1691208
29: -18.3540745, 10.1912298, -18.1967468, 10.1758862, -26.9430847, 26.8145981
30: -14.8668213, 13.3494921, -14.7986908, 13.3206863, -27.4294357, 27.3716736
31: -17.5580559, 2.9539719, -17.5203571, 2.9196391, -20.4776955, 20.4743290
32: -13.3836403, 13.6593189, -13.3139191, 13.6415367, -27.0251770, 26.9682083
33: -32.1217995, 6.9730496, -32.0870743, 6.8744950, -37.1108322, 37.1705475
34: -25.7384682, 7.8854451, -25.7124863, 7.8360262, -31.1347809, 31.1580582
35: -19.9922218, 11.5322437, -19.9544601, 11.4833269, -30.6346893, 30.6571350
36: -20.0491753, 15.7131042, -19.9854660, 15.6957645, -33.0844803, 33.0330124
37: -28.6355972, 7.5077581, -28.5753613, 7.4308667, -33.4420853, 33.4579926
38: -25.0854206, 16.9443588, -25.0292797, 16.8440571, -41.5502777, 41.6006546
39: -36.4789619, 8.7730417, -36.4457130, 8.6734524, -44.0238953, 44.0992355
40: -31.1935120, 1.1598697, -31.1549511, 1.0847607, -27.7738800, 27.7664032
41: -18.2756672, 11.9269085, -18.2409763, 11.9157467, -26.3498764, 26.3122482
42: -11.4109087, 11.5469484, -11.3694496, 11.5210133, -22.9319229, 22.9163971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1473
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1311
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1399

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9884045, upper bound: 16.9884045
time: 32.11 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434
time: 35.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 70.59 seconds
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9164283, upper bound: 16.9401857
IS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9165777, upper bound: 17.0582654
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9164283, upper bound: 16.9401857
IS_A2_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9759248, upper bound: 17.0582654
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9290624, upper bound: 16.9884045
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9884045, upper bound: 16.9884045
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 70.59
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -31.3843880, 1.6737804, -31.3221931, 1.5492415, -32.9336281, 32.9959717
1: -14.7757521, 4.6753392, -14.7274113, 4.5766721, -19.3524246, 19.4027500
2: -19.7372971, 1.6909714, -19.7069435, 1.6064434, -19.6573677, 19.7123489
3: -19.3886871, 1.2213588, -19.3543777, 1.1385970, -19.9699020, 20.0096588
4: -26.5552750, 0.2408433, -26.5081902, 0.1122026, -24.7408371, 24.8275070
5: -23.2522697, 1.9988019, -23.2193375, 1.8910816, -21.6948471, 21.7688332
6: -18.7867813, 6.9234142, -18.6810684, 6.8766861, -24.8884659, 24.7910500
7: -24.9383659, 0.9476995, -24.8842487, 0.8351531, -22.5225525, 22.5823021
8: -35.0442047, -2.0687442, -34.9837837, -2.2150288, -29.9575653, 30.0498428
9: -13.1071367, 11.8167343, -13.0688791, 11.7379799, -24.8451157, 24.8856125
10: -12.9300652, 18.1056633, -12.8794556, 17.9844971, -30.9145622, 30.9851189
11: -12.4118910, 14.5783339, -12.2322369, 14.5496626, -25.7925873, 25.6319008
12: -0.4350395, 32.8879929, -0.1950045, 32.8249817, -30.7532349, 30.5658951
13: -21.0392151, 19.0691452, -20.9643898, 19.0040455, -39.4128418, 39.4041367
14: -45.1020737, -0.7157464, -45.0025520, -0.7519794, -34.3780060, 34.2598419
15: -22.5750160, -0.3658981, -22.5067139, -0.5200553, -20.7138939, 20.8009377
16: -18.1649685, 7.2551451, -18.0728951, 7.1669054, -25.3318748, 25.3280411
17: -34.9183273, 10.8402805, -34.5281143, 10.7802382, -43.4528046, 43.1162720
18: -12.0074825, 14.3750496, -11.9468002, 14.3191309, -26.3266144, 26.3218498
19: -15.8295412, 2.0011013, -15.7531710, 1.9750354, -17.8045769, 17.7542725
20: -10.9531898, 5.0985289, -10.8904381, 5.0351295, -15.9883194, 15.9889669
21: -12.1605015, 7.5431252, -12.0310078, 7.5054622, -19.6659641, 19.5741329
22: -14.0777283, 7.2898169, -13.8896971, 7.2398643, -21.3175926, 21.1795139
23: -12.7093544, 8.1237354, -12.6484976, 8.0882044, -19.2161064, 19.1845360
24: -18.1631336, 7.0402880, -18.1014652, 6.9707370, -25.1338711, 25.1417542
25: -10.9165878, 8.3659744, -10.8475895, 8.3013153, -19.2179031, 19.2135639
26: -14.2896957, 15.5979919, -14.1353283, 15.5874500, -29.8771458, 29.7333202
27: -24.8337440, 7.0889387, -24.7675190, 7.0683236, -31.9020672, 31.8564568
28: -14.9347105, 9.0266514, -14.8454294, 9.0042477, -23.2066498, 23.1226311
29: -18.3469620, 10.1744461, -18.1106968, 10.1375103, -26.8972931, 26.6921844
30: -14.8627987, 13.3409672, -14.7668705, 13.2964363, -27.4009552, 27.3159637
31: -17.5479889, 2.9514928, -17.4885330, 2.8948579, -20.4428463, 20.4400253
32: -13.3765059, 13.6445341, -13.2563057, 13.5928345, -26.9693413, 26.8847961
33: -32.1034546, 6.9689460, -32.0330200, 6.8223600, -37.0242767, 37.1089020
34: -25.7273293, 7.8818378, -25.6724834, 7.7970419, -31.0741577, 31.1109085
35: -19.9830894, 11.5302305, -19.9190178, 11.4541798, -30.5821381, 30.6200485
36: -20.0387077, 15.7095871, -19.9356613, 15.6772614, -33.0492935, 32.9760361
37: -28.6155739, 7.5047250, -28.5064850, 7.3894634, -33.3713608, 33.3832397
38: -25.0649757, 16.9414902, -24.9659996, 16.8046227, -41.4662781, 41.5387650
39: -36.4542313, 8.7701006, -36.3934822, 8.6343899, -43.9374847, 44.0429459
40: -31.1788692, 1.1560116, -31.1168594, 1.0445700, -27.6689758, 27.7319489
41: -18.2682800, 11.9225168, -18.2127476, 11.9013882, -26.3164597, 26.2714767
42: -11.4053612, 11.5367432, -11.3329992, 11.5056725, -22.9110336, 22.8697433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=193, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.7983991, upper bound: 17.0580986
time: 30.28 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9165776, upper bound: 17.0582655
time: 33.37 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -31.3942261, 1.6765003, -31.3638229, 1.6037188, -32.9979439, 33.0403214
1: -14.7779818, 4.6768818, -14.7492962, 4.6187658, -19.3967476, 19.4261780
2: -19.7456417, 1.6930778, -19.7361889, 1.6411383, -19.7025986, 19.7416611
3: -19.3939266, 1.2231555, -19.3784313, 1.1792202, -20.0148315, 20.0294418
4: -26.5686703, 0.2427125, -26.5546532, 0.1629047, -24.8054810, 24.8702850
5: -23.2582169, 2.0012677, -23.2437630, 1.9398873, -21.7489815, 21.7840424
6: -18.7894592, 6.9348555, -18.7326794, 6.9195580, -24.9328842, 24.8668289
7: -24.9423218, 0.9497457, -24.9128170, 0.8818817, -22.5767441, 22.6044579
8: -35.0616074, -2.0656223, -35.0436668, -2.1513877, -30.0405502, 30.1026535
9: -13.1096096, 11.8196716, -13.0891132, 11.7747555, -24.8843651, 24.9087849
10: -12.9334249, 18.1091118, -12.9073906, 18.0366554, -30.9700813, 31.0165024
11: -12.4149075, 14.5817595, -12.2987423, 14.5685129, -25.8148880, 25.7068558
12: -0.4391479, 32.9062843, -0.3014002, 32.8883667, -30.8065109, 30.6919746
13: -21.0423031, 19.0770931, -21.0050125, 19.0451317, -39.4589767, 39.4590073
14: -45.1057434, -0.7115693, -45.0456772, -0.7284327, -34.4038010, 34.3441010
15: -22.5862808, -0.3645649, -22.5583916, -0.4566584, -20.7883682, 20.8474503
16: -18.1673698, 7.2590060, -18.1067619, 7.2097988, -25.3771687, 25.3657684
17: -34.9224091, 10.8543301, -34.6711006, 10.8310232, -43.5057449, 43.2736282
18: -12.0115051, 14.3774929, -11.9785662, 14.3466911, -26.3581963, 26.3560600
19: -15.8329601, 2.0019996, -15.7889748, 1.9865811, -17.8195419, 17.7909737
20: -10.9575777, 5.0999269, -10.9248991, 5.0589523, -16.0165291, 16.0248260
21: -12.1643448, 7.5442896, -12.0872602, 7.5226107, -19.6869545, 19.6315498
22: -14.0821266, 7.2950315, -13.9638577, 7.2695880, -21.3517151, 21.2588882
23: -12.7122917, 8.1249247, -12.6783361, 8.1074667, -19.2395935, 19.2180862
24: -18.1705055, 7.0424590, -18.1389904, 7.0064635, -25.1769695, 25.1814499
25: -10.9204769, 8.3674889, -10.8832703, 8.3289938, -19.2494698, 19.2507591
26: -14.2941113, 15.5989990, -14.2044067, 15.5949326, -29.8890438, 29.8034058
27: -24.8388195, 7.0916271, -24.7981758, 7.0850410, -31.9238605, 31.8898029
28: -14.9372425, 9.0305729, -14.8812857, 9.0221558, -23.2272644, 23.1680489
29: -18.3509598, 10.1853046, -18.1967468, 10.1758862, -26.9399261, 26.7925262
30: -14.8649836, 13.3444166, -14.7986908, 13.3206863, -27.4275513, 27.3627701
31: -17.5516548, 2.9527979, -17.5203571, 2.9196391, -20.4712944, 20.4731560
32: -13.3794327, 13.6573801, -13.3139191, 13.6415367, -27.0209694, 26.9634781
33: -32.1149406, 6.9714441, -32.0870743, 6.8744950, -37.0947113, 37.1685867
34: -25.7361584, 7.8839216, -25.7124863, 7.8360262, -31.1350403, 31.1551132
35: -19.9882107, 11.5310841, -19.9544601, 11.4833269, -30.6306915, 30.6562042
36: -20.0427589, 15.7124729, -19.9854660, 15.6957645, -33.0700531, 33.0321579
37: -28.6291542, 7.5058441, -28.5753613, 7.4308667, -33.4314880, 33.4560852
38: -25.0762310, 16.9428692, -25.0292797, 16.8440571, -41.5401154, 41.6015472
39: -36.4672241, 8.7714825, -36.4457130, 8.6734524, -44.0075531, 44.0974350
40: -31.1868839, 1.1578736, -31.1549511, 1.0847607, -27.7222595, 27.7643433
41: -18.2724991, 11.9250755, -18.2409763, 11.9157467, -26.3352051, 26.3103027
42: -11.4089317, 11.5390120, -11.3694496, 11.5210133, -22.9299450, 22.9084625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1473
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1311
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1384

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8577666, upper bound: 17.0580986
time: 32.32 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9759247, upper bound: 17.0582655
time: 32.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 67.18 seconds
IS_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 67.18
Output dim: 12, lower bound: -16.7983991, upper bound: 17.0580986
IS_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 67.18
Output dim: 12, lower bound: -16.9165776, upper bound: 17.0582655
IS_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 67.18
Output dim: 12, lower bound: -16.8577666, upper bound: 17.0580986
IS_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 67.18
Output dim: 12, lower bound: -16.9759247, upper bound: 17.0582655

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 94.81 + 759.70 = 854.51 seconds
