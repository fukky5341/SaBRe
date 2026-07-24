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
execution time: IAR + RelationalAnalysis = 2.94 + 90.27 = 93.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -17.0952923, upper bound: 17.0952923

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1658

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0315090
time: 26.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0861274
time: 33.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 60.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 60.88
Output dim: 12, lower bound: -17.0861276, upper bound: 17.0315090
IS_A2, status: Status.UNKNOWN, split count: 1, time: 60.88
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

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1399

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
time: 29.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
time: 35.93 seconds

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

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1399

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
time: 32.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
time: 34.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 69.39 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 69.39
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 69.39
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0188180
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 69.39
Output dim: 12, lower bound: -16.9910876, upper bound: 17.0734299
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 69.39
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

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9884046, upper bound: 17.0114178
time: 41.25 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9884046, upper bound: 17.0707432
time: 29.72 seconds

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

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1399

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0734301, upper bound: 16.9910874
time: 40.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -17.0734301, upper bound: 17.0734299
time: 41.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 85.23 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 85.23
Output dim: 12, lower bound: -16.9884046, upper bound: 17.0114178
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 85.23
Output dim: 12, lower bound: -16.9884046, upper bound: 17.0707432
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 85.23
Output dim: 12, lower bound: -17.0734301, upper bound: 16.9910874
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 85.23
Output dim: 12, lower bound: -17.0734301, upper bound: 17.0734299

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -31.3878384, 1.6757054, -31.3509712, 1.6049395, -32.9927788, 33.0266762
1: -14.7758160, 4.6760149, -14.7481213, 4.6175838, -19.3934002, 19.4241371
2: -19.7325001, 1.6918364, -19.6962509, 1.6414127, -19.6957321, 19.7160759
3: -19.3876839, 1.2225671, -19.3623447, 1.1811948, -19.9947472, 20.0022240
4: -26.5488186, 0.2413211, -26.4875565, 0.1681676, -24.7902756, 24.8139648
5: -23.2438431, 2.0001132, -23.2036438, 1.9451869, -21.7377167, 21.7819557
6: -18.7831039, 6.9306402, -18.7113132, 6.9124775, -24.9256516, 24.8348122
7: -24.9272919, 0.9489446, -24.8664303, 0.8836856, -22.5762558, 22.6001053
8: -35.0511169, -2.0686259, -35.0171700, -2.1577849, -30.0180359, 30.0900650
9: -13.1075077, 11.8155537, -13.0860233, 11.7629337, -24.8704414, 24.9015770
10: -12.9286976, 18.0930901, -12.9115009, 17.9758472, -30.9045448, 31.0045910
11: -12.4137058, 14.5740929, -12.3041925, 14.5440140, -25.8000183, 25.7122040
12: -0.4359112, 32.8772202, -0.2995329, 32.7912369, -30.7625961, 30.6784592
13: -21.0371246, 19.0737782, -20.9922581, 19.0491886, -39.4627533, 39.4374008
14: -45.1040497, -0.7276151, -45.0505066, -0.7858891, -34.4011002, 34.3428574
15: -22.5803871, -0.3654065, -22.5473289, -0.4523702, -20.7770462, 20.8445396
16: -18.1649094, 7.2573204, -18.1067734, 7.2054749, -25.3703842, 25.3640938
17: -34.9210625, 10.8323708, -34.6820831, 10.7545300, -43.4456940, 43.2774429
18: -12.0068445, 14.3745079, -11.9684486, 14.3406620, -26.3475075, 26.3429565
19: -15.8261719, 2.0009854, -15.7639847, 1.9866335, -17.8128052, 17.7649708
20: -10.9472065, 5.0987606, -10.8914585, 5.0571585, -16.0043640, 15.9902191
21: -12.1619911, 7.5433226, -12.0829916, 7.5254378, -19.6874294, 19.6263142
22: -14.0796900, 7.2920761, -13.9591722, 7.2647595, -21.3444500, 21.2512474
23: -12.7095261, 8.1225853, -12.6797199, 8.1000376, -19.2308006, 19.2196999
24: -18.1683197, 7.0401611, -18.1341400, 7.0034037, -25.1717224, 25.1743011
25: -10.9166365, 8.3645668, -10.8844376, 8.3196115, -19.2362480, 19.2490044
26: -14.2926207, 15.5925045, -14.2061329, 15.5704212, -29.8630409, 29.7986374
27: -24.8232193, 7.0886316, -24.7531776, 7.0756006, -31.8988190, 31.8418083
28: -14.9359655, 9.0280704, -14.8830376, 9.0225296, -23.2278900, 23.1650429
29: -18.3489017, 10.1732016, -18.1988792, 10.1347437, -26.9134026, 26.7962646
30: -14.8640079, 13.3352499, -14.8017273, 13.2900181, -27.4052200, 27.3560715
31: -17.5408688, 2.9512424, -17.4795876, 2.9206281, -20.4614964, 20.4308300
32: -13.3731880, 13.6539669, -13.2923450, 13.6407862, -27.0139732, 26.9357643
33: -32.1001587, 6.9700651, -32.0468445, 6.8798518, -37.0918350, 37.1345291
34: -25.7309799, 7.8816671, -25.7091064, 7.8313241, -31.1146393, 31.1490631
35: -19.9805832, 11.5306644, -19.9317780, 11.4871321, -30.6177292, 30.6424408
36: -20.0314026, 15.7096844, -19.9435062, 15.7038736, -33.0797882, 32.9901199
37: -28.6143188, 7.5053596, -28.5384693, 7.4347458, -33.4239960, 33.4221954
38: -25.0597382, 16.9410877, -24.9766273, 16.8531437, -41.5138550, 41.5539703
39: -36.4450150, 8.7706232, -36.3698654, 8.6846447, -43.9909515, 44.0345306
40: -31.1699753, 1.1571364, -31.1051407, 1.0911565, -27.7381439, 27.7543335
41: -18.2616920, 11.9213037, -18.2086487, 11.9101238, -26.3380432, 26.2856827
42: -11.4065943, 11.5307188, -11.3690910, 11.4849997, -22.8915939, 22.8998108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9795526, upper bound: 17.0202401
time: 29.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9795526, upper bound: 17.0618849
time: 32.98 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -31.3797894, 1.6774125, -31.3682785, 1.6044755, -32.9842644, 33.0456924
1: -14.7759619, 4.6755528, -14.7516212, 4.6192279, -19.3951893, 19.4271736
2: -19.7044716, 1.6931438, -19.7396984, 1.6417475, -19.6733856, 19.7596550
3: -19.3771400, 1.2247281, -19.3807602, 1.1804240, -19.9828529, 20.0273209
4: -26.5001068, 0.2477889, -26.5588779, 0.1634445, -24.7437744, 24.8916626
5: -23.2163162, 2.0058463, -23.2487411, 1.9418206, -21.7353706, 21.8114662
6: -18.7678814, 6.9265537, -18.7334251, 6.9229565, -24.9169922, 24.8486519
7: -24.8948936, 0.9513240, -24.9165382, 0.8824444, -22.5622940, 22.6493225
8: -35.0333977, -2.0723462, -35.0488052, -2.1505318, -30.0176010, 30.1186447
9: -13.1060734, 11.8069096, -13.0904016, 11.7772102, -24.8832836, 24.8973122
10: -12.9362411, 18.0478477, -12.9110670, 18.0380135, -30.9742546, 30.9589157
11: -12.4201050, 14.5566998, -12.2995138, 14.5702744, -25.8310661, 25.6919861
12: -0.4367132, 32.8073692, -0.3030758, 32.8934860, -30.8649750, 30.6351357
13: -21.0288734, 19.0795822, -21.0065613, 19.0492706, -39.4540100, 39.4625320
14: -45.1101189, -0.7698252, -45.0471497, -0.7264042, -34.4540100, 34.3131485
15: -22.5736122, -0.3604250, -22.5627670, -0.4562550, -20.7789192, 20.8584099
16: -18.1668663, 7.2542458, -18.1084042, 7.2110462, -25.3779125, 25.3626499
17: -34.9330254, 10.7764206, -34.6721153, 10.8350964, -43.5422211, 43.2124329
18: -12.0007553, 14.3710232, -11.9814482, 14.3478546, -26.3486099, 26.3524704
19: -15.8077488, 2.0019412, -15.7899675, 1.9868848, -17.7946339, 17.7919083
20: -10.9236240, 5.0979471, -10.9267826, 5.0594926, -15.9831161, 16.0247307
21: -12.1597214, 7.5468826, -12.0883751, 7.5231185, -19.6828403, 19.6352577
22: -14.0769386, 7.2894945, -13.9651937, 7.2715616, -21.3484993, 21.2546883
23: -12.7125330, 8.1172285, -12.6812935, 8.1082096, -19.2419739, 19.2157516
24: -18.1650982, 7.0389814, -18.1409988, 7.0075569, -25.1726551, 25.1799812
25: -10.9198866, 8.3578882, -10.8883934, 8.3297148, -19.2496014, 19.2462807
26: -14.2954111, 15.5742731, -14.2056160, 15.5955820, -29.8909931, 29.7798882
27: -24.7930336, 7.0818014, -24.8004951, 7.0862150, -31.8792496, 31.8822975
28: -14.9386673, 9.0299883, -14.8822088, 9.0248232, -23.2321167, 23.1659622
29: -18.3527641, 10.1432028, -18.1978149, 10.1785898, -26.9627151, 26.7667694
30: -14.8678312, 13.3129730, -14.7992973, 13.3228722, -27.4380341, 27.3347244
31: -17.5105820, 2.9536042, -17.5213928, 2.9201593, -20.4307404, 20.4749966
32: -13.3574533, 13.6553392, -13.3151531, 13.6450844, -27.0025368, 26.9595871
33: -32.0728226, 6.9765773, -32.0929565, 6.8751612, -37.0592957, 37.1898270
34: -25.7316971, 7.8789539, -25.7154140, 7.8367271, -31.1215973, 31.1489029
35: -19.9648361, 11.5348101, -19.9566841, 11.4835453, -30.6032791, 30.6652832
36: -20.0004387, 15.7194433, -19.9868202, 15.6994667, -33.0446625, 33.0480347
37: -28.5902500, 7.5096021, -28.5814819, 7.4312129, -33.3937073, 33.4713287
38: -25.0223122, 16.9513931, -25.0334244, 16.8455353, -41.4700317, 41.6124268
39: -36.3898239, 8.7824907, -36.4501266, 8.6738491, -43.9286499, 44.1216049
40: -31.1355324, 1.1640563, -31.1598892, 1.0854383, -27.7056503, 27.8236542
41: -18.2396812, 11.9177961, -18.2424622, 11.9204254, -26.3247070, 26.3196945
42: -11.4082298, 11.5027609, -11.3703976, 11.5219593, -22.9301891, 22.8731575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9290624, upper bound: 16.9884045
time: 33.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9884045, upper bound: 16.9884045
time: 33.41 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -31.3956757, 1.6767182, -31.3682785, 1.6044755, -33.0001526, 33.0449982
1: -14.7787352, 4.6770229, -14.7516212, 4.6192279, -19.3979626, 19.4286442
2: -19.7467842, 1.6932571, -19.7396984, 1.6417475, -19.7044296, 19.7496529
3: -19.3947411, 1.2235472, -19.3807602, 1.1804240, -20.0172386, 20.0412636
4: -26.5700035, 0.2428722, -26.5588779, 0.1634445, -24.8073959, 24.8804626
5: -23.2598133, 2.0018840, -23.2487411, 1.9418206, -21.7530785, 21.8040085
6: -18.7897263, 6.9359565, -18.7334251, 6.9229565, -24.9384613, 24.8631325
7: -24.9436493, 0.9499116, -24.9165382, 0.8824444, -22.5792389, 22.6200294
8: -35.0631943, -2.0653553, -35.0488052, -2.1505318, -30.0430145, 30.1201324
9: -13.1100311, 11.8203640, -13.0904016, 11.7772102, -24.8872414, 24.9107666
10: -12.9346256, 18.1095467, -12.9110670, 18.0380135, -30.9726391, 31.0206146
11: -12.4151831, 14.5823326, -12.2995138, 14.5702744, -25.8181305, 25.7081299
12: -0.4397106, 32.9079590, -0.3030758, 32.8934860, -30.8280258, 30.6952515
13: -21.0428123, 19.0784454, -21.0065613, 19.0492706, -39.4686127, 39.4620438
14: -45.1062355, -0.7109697, -45.0471497, -0.7264042, -34.4122009, 34.3338852
15: -22.5877075, -0.3644352, -22.5627670, -0.4562550, -20.7905197, 20.8583488
16: -18.1679115, 7.2594223, -18.1084042, 7.2110462, -25.3789577, 25.3678265
17: -34.9227371, 10.8556528, -34.6721153, 10.8350964, -43.5127869, 43.2761688
18: -12.0124874, 14.3778381, -11.9814482, 14.3478546, -26.3603420, 26.3592873
19: -15.8332882, 2.0020957, -15.7899675, 1.9868848, -17.8201733, 17.7920628
20: -10.9582338, 5.1001039, -10.9267826, 5.0594926, -16.0177269, 16.0268860
21: -12.1647139, 7.5444098, -12.0883751, 7.5231185, -19.6878319, 19.6327858
22: -14.0825644, 7.2956667, -13.9651937, 7.2715616, -21.3541260, 21.2608604
23: -12.7131977, 8.1251631, -12.6812935, 8.1082096, -19.2412109, 19.2216263
24: -18.1712170, 7.0428038, -18.1409988, 7.0075569, -25.1787739, 25.1838036
25: -10.9221163, 8.3677197, -10.8883934, 8.3297148, -19.2518311, 19.2561131
26: -14.2945032, 15.5992336, -14.2056160, 15.5955820, -29.8900852, 29.8048496
27: -24.8395977, 7.0920081, -24.8004951, 7.0862150, -31.9258118, 31.8925037
28: -14.9375286, 9.0314245, -14.8822088, 9.0248232, -23.2308502, 23.1682739
29: -18.3513298, 10.1861877, -18.1978149, 10.1785898, -26.9438477, 26.7942123
30: -14.8651905, 13.3450985, -14.7992973, 13.3228722, -27.4323196, 27.3621140
31: -17.5520077, 2.9529657, -17.5213928, 2.9201593, -20.4721680, 20.4743576
32: -13.3798265, 13.6585102, -13.3151531, 13.6450844, -27.0249100, 26.9625778
33: -32.1169281, 6.9716640, -32.0929565, 6.8751612, -37.0973816, 37.1761703
34: -25.7371140, 7.8841147, -25.7154140, 7.8367271, -31.1338806, 31.1612396
35: -19.9889259, 11.5311518, -19.9566841, 11.4835453, -30.6268234, 30.6622009
36: -20.0432339, 15.7137623, -19.9868202, 15.6994667, -33.0797119, 33.0349731
37: -28.6311684, 7.5059857, -28.5814819, 7.4312129, -33.4329224, 33.4631729
38: -25.0776024, 16.9433022, -25.0334244, 16.8455353, -41.5324402, 41.6086044
39: -36.4686050, 8.7715778, -36.4501266, 8.6738491, -44.0039368, 44.1062317
40: -31.1885872, 1.1580796, -31.1598892, 1.0854383, -27.7253342, 27.7773514
41: -18.2729645, 11.9265509, -18.2424622, 11.9204254, -26.3433228, 26.3130264
42: -11.4092455, 11.5393391, -11.3703976, 11.5219593, -22.9312057, 22.9097366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
time: 33.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434
time: 44.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 80.85 seconds
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9795526, upper bound: 17.0202401
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9795526, upper bound: 17.0618849
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9290624, upper bound: 16.9884045
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9884045, upper bound: 16.9884045
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9290624, upper bound: 17.0707434
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 80.85
Output dim: 12, lower bound: -16.9884045, upper bound: 17.0707434

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -31.3852196, 1.6698475, -31.4840698, 1.6114783, -32.9966965, 33.1539154
1: -14.7747784, 4.6734877, -14.8198652, 4.6227951, -19.3975735, 19.4933529
2: -19.7312508, 1.6879048, -19.7641277, 1.6456609, -19.6930008, 19.7831345
3: -19.3857613, 1.2167931, -19.4790688, 1.1912589, -19.9909515, 20.1146927
4: -26.5474472, 0.2378678, -26.5826607, 0.1960149, -24.8134232, 24.9088287
5: -23.2423592, 1.9938347, -23.3279686, 1.9585869, -21.7342606, 21.9011497
6: -18.7761784, 6.9274540, -18.7271385, 6.9784994, -24.9972305, 24.8460007
7: -24.9261723, 0.9435172, -24.9773865, 0.8927703, -22.5770607, 22.7136841
8: -35.0492401, -2.0731964, -35.1005325, -2.1436734, -30.0239792, 30.1685638
9: -13.1054401, 11.8122053, -13.1329851, 11.7835331, -24.8889732, 24.9451904
10: -12.9259787, 18.0888710, -12.9732676, 18.0065193, -30.9324989, 31.0621376
11: -12.4103050, 14.5725021, -12.3557520, 14.5909615, -25.8401947, 25.7698250
12: -0.4273434, 32.8761597, -0.3354502, 32.8818626, -30.8305511, 30.7195969
13: -21.0342636, 19.0694427, -21.1224499, 19.0579071, -39.4671478, 39.5675583
14: -45.0998421, -0.7338095, -45.2580414, -0.7909870, -34.3761749, 34.5470123
15: -22.5768318, -0.3728628, -22.5810719, -0.4327765, -20.7936478, 20.8738022
16: -18.1603279, 7.2547178, -18.1391869, 7.2302942, -25.3906212, 25.3939056
17: -34.9173584, 10.8284969, -34.9137917, 10.7637024, -43.4469147, 43.5095062
18: -11.9981174, 14.3721581, -11.9743118, 14.4303188, -26.4284363, 26.3464699
19: -15.8249683, 1.9996545, -15.7894373, 2.0037968, -17.8287659, 17.7890911
20: -10.9458370, 5.0961299, -10.9312878, 5.0680494, -16.0138855, 16.0274181
21: -12.1602345, 7.5420399, -12.1212587, 7.5457106, -19.7059441, 19.6632996
22: -14.0764151, 7.2838774, -14.0091724, 7.2717667, -21.3481827, 21.2930489
23: -12.7063370, 8.1211329, -12.6947136, 8.1455774, -19.2830162, 19.2279053
24: -18.1643677, 7.0382066, -18.1445866, 7.0599527, -25.2243195, 25.1827927
25: -10.9140081, 8.3628578, -10.9028492, 8.3463068, -19.2603149, 19.2657070
26: -14.2846270, 15.5910988, -14.2208195, 15.6541958, -29.9388237, 29.8119183
27: -24.8190041, 7.0866642, -24.7650700, 7.1473694, -31.9663734, 31.8517342
28: -14.9339838, 9.0266352, -14.8993406, 9.0520678, -23.2701874, 23.1690712
29: -18.3463612, 10.1688004, -18.2643661, 10.1394138, -26.9191818, 26.8652840
30: -14.8608513, 13.3330612, -14.8216391, 13.3534393, -27.4689026, 27.3746109
31: -17.5395756, 2.9471774, -17.5086479, 2.9300675, -20.4696426, 20.4558258
32: -13.3670254, 13.6514463, -13.3172321, 13.6546249, -27.0216503, 26.9570389
33: -32.0959244, 6.9678869, -32.0713806, 6.9856019, -37.1994324, 37.1548462
34: -25.7271042, 7.8805141, -25.7249470, 7.8990846, -31.1726227, 31.1645737
35: -19.9773026, 11.5295019, -19.9607086, 11.5300541, -30.6576385, 30.6667786
36: -20.0275497, 15.7088480, -19.9793186, 15.7319126, -33.1035309, 33.0230789
37: -28.6075516, 7.5036993, -28.5626602, 7.6230392, -33.6078415, 33.4338608
38: -25.0565872, 16.9394436, -25.0251236, 16.8804150, -41.5212555, 41.6187592
39: -36.4408150, 8.7694130, -36.4030304, 8.7555714, -44.0605621, 44.0634460
40: -31.1653519, 1.1548576, -31.1208725, 1.2552867, -27.8999100, 27.7652283
41: -18.2575493, 11.9192514, -18.2193108, 12.0305052, -26.4548035, 26.2907639
42: -11.3992043, 11.5284309, -11.3836880, 11.5692358, -22.9684410, 22.9121189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1416

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9669258, upper bound: 16.9313232
time: 34.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9670729, upper bound: 17.0494067
time: 38.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

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

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.8786211, upper bound: 17.0618852
time: 30.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.8786211, upper bound: 17.0618851
time: 34.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2

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

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9379279, upper bound: 17.0618852
time: 51.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -16.9795522, upper bound: 17.0618852
time: 35.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 89.67 seconds
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.9669258, upper bound: 16.9313232
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.9670729, upper bound: 17.0494067
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.8786211, upper bound: 17.0618852
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.8786211, upper bound: 17.0618851
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.9379279, upper bound: 17.0618852
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 89.67
Output dim: 12, lower bound: -16.9795522, upper bound: 17.0618852

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -31.3739586, 1.6009779, -31.3195305, 1.5301409, -32.9040985, 32.9205093
1: -14.7696943, 4.6471157, -14.7258539, 4.5695405, -19.3392353, 19.3729706
2: -19.7322712, 1.6413727, -19.7056694, 1.5935340, -19.6375122, 19.6598930
3: -19.3797455, 1.1538479, -19.3521023, 1.1210380, -19.9433212, 19.9437027
4: -26.5468483, 0.1883621, -26.5060081, 0.0987267, -24.7190247, 24.7731857
5: -23.2447395, 1.9216609, -23.2174072, 1.8709767, -21.6666794, 21.6887932
6: -18.7521839, 6.9057894, -18.6711807, 6.8720570, -24.8476257, 24.7641563
7: -24.9305706, 0.8870702, -24.8822498, 0.8194070, -22.4992676, 22.5190468
8: -35.0356178, -2.1321774, -34.9815826, -2.2311659, -29.9359741, 29.9945679
9: -13.0956287, 11.7965059, -13.0659409, 11.7328405, -24.8284683, 24.8624458
10: -12.9179430, 18.0839329, -12.8762703, 17.9789124, -30.8968544, 30.9602032
11: -12.3976851, 14.5706005, -12.2285118, 14.5476665, -25.7758980, 25.6197662
12: -0.3816566, 32.8764954, -0.1811047, 32.8220863, -30.7029648, 30.5372353
13: -21.0207901, 19.0021744, -20.9596577, 18.9869537, -39.3767090, 39.3334427
14: -45.0860596, -0.8168035, -44.9983482, -0.7776580, -34.3343811, 34.1530762
15: -22.5597458, -0.3914208, -22.5026932, -0.5268722, -20.6917801, 20.7713890
16: -18.1546478, 7.2420864, -18.0702229, 7.1635685, -25.3182163, 25.3123093
17: -34.9031792, 10.7794495, -34.5241699, 10.7646170, -43.4213715, 43.0508652
18: -11.9653568, 14.3626900, -11.9359150, 14.3160105, -26.2813683, 26.2986050
19: -15.8221169, 1.9920166, -15.7511683, 1.9727690, -17.7948856, 17.7431850
20: -10.9466610, 5.0873971, -10.8887463, 5.0322356, -15.9788971, 15.9761429
21: -12.1482534, 7.5346360, -12.0277548, 7.5033112, -19.6515656, 19.5623913
22: -14.0631161, 7.2789116, -13.8857203, 7.2371130, -21.3002281, 21.1646309
23: -12.6918640, 8.1148434, -12.6439552, 8.0859165, -19.1903229, 19.1699867
24: -18.1385880, 7.0317554, -18.0949059, 6.9685917, -25.1071796, 25.1266613
25: -10.9029484, 8.3594217, -10.8437271, 8.2996311, -19.2025795, 19.2031479
26: -14.2266016, 15.5899601, -14.1187172, 15.5854120, -29.8120136, 29.7086773
27: -24.7997570, 7.0774775, -24.7586441, 7.0654292, -31.8651867, 31.8361206
28: -14.9229832, 9.0161524, -14.8422298, 9.0015697, -23.1824951, 23.1052170
29: -18.3368759, 10.1642208, -18.1079941, 10.1349211, -26.8829727, 26.6773071
30: -14.8387537, 13.3317652, -14.7606220, 13.2940884, -27.3755493, 27.3001556
31: -17.5400810, 2.9331865, -17.4864540, 2.8902612, -20.4303417, 20.4196396
32: -13.3624859, 13.6237249, -13.2526846, 13.5873337, -26.9498196, 26.8560104
33: -32.0450668, 6.9582829, -32.0181313, 6.8195963, -36.9646378, 37.0828476
34: -25.6987343, 7.8741417, -25.6648636, 7.7950063, -31.0427551, 31.0942764
35: -19.9613686, 11.5244408, -19.9131355, 11.4526072, -30.5579987, 30.6077499
36: -20.0103493, 15.7042522, -19.9282684, 15.6758518, -33.0192261, 32.9624023
37: -28.5173817, 7.4982491, -28.4813938, 7.3877778, -33.2708359, 33.3512192
38: -25.0394897, 16.9332733, -24.9594421, 16.8024330, -41.4315491, 41.5117035
39: -36.4175110, 8.7652149, -36.3839073, 8.6331921, -43.9009399, 44.0283661
40: -31.1036644, 1.1412573, -31.0980415, 1.0407867, -27.5885239, 27.6974716
41: -18.2016659, 11.9123001, -18.1960678, 11.8987341, -26.2463684, 26.2432442
42: -11.3570318, 11.5233364, -11.3208942, 11.5022192, -22.8592510, 22.8442307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=193, inp2_unstable=193, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.7477054, upper bound: 17.0492606
time: 37.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8661213, upper bound: 17.0494071
time: 31.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -31.5175018, 1.6804872, -31.3196335, 1.5433764, -33.0608788, 33.0001221
1: -14.8475285, 4.6805148, -14.7264061, 4.5741267, -19.4216557, 19.4069214
2: -19.8051224, 1.6951571, -19.7056961, 1.6024640, -19.7244034, 19.7096062
3: -19.5053978, 1.2314956, -19.3524647, 1.1328187, -20.0823822, 20.0058861
4: -26.6503735, 0.2686687, -26.5067749, 0.1087503, -24.8356781, 24.8505859
5: -23.3765678, 2.0122461, -23.2178612, 1.8847961, -21.8140564, 21.7654305
6: -18.8024616, 6.9894719, -18.6743393, 6.8734388, -24.8996658, 24.8626633
7: -25.0492630, 0.9568114, -24.8831367, 0.8297033, -22.6361160, 22.5830688
8: -35.1274605, -2.0546389, -34.9819565, -2.2195716, -30.0361176, 30.0557556
9: -13.1541004, 11.8373241, -13.0668354, 11.7346611, -24.8887615, 24.9041595
10: -12.9918079, 18.1364632, -12.8767557, 17.9802570, -30.9720650, 31.0132179
11: -12.4636555, 14.6252880, -12.2288303, 14.5480642, -25.8505936, 25.6721001
12: -0.4710026, 32.9786301, -0.1864233, 32.8238983, -30.7943726, 30.6338348
13: -21.1693802, 19.0779152, -20.9615135, 18.9997292, -39.5430145, 39.4084930
14: -45.3097992, -0.7209635, -44.9983253, -0.7581291, -34.5824203, 34.2350006
15: -22.6088753, -0.3463130, -22.5031281, -0.5275044, -20.7432251, 20.8174400
16: -18.1975288, 7.2799664, -18.0683022, 7.1643000, -25.3618279, 25.3482685
17: -35.1502304, 10.8493776, -34.5244255, 10.7763424, -43.6851425, 43.1174469
18: -12.0131435, 14.4647579, -11.9380779, 14.3168106, -26.3299541, 26.4028358
19: -15.8550463, 2.0182693, -15.7519665, 1.9737341, -17.8287811, 17.7702351
20: -10.9930668, 5.1093898, -10.8890877, 5.0325217, -16.0255890, 15.9984779
21: -12.1989136, 7.5634193, -12.0292883, 7.5041690, -19.7030830, 19.5927086
22: -14.1278429, 7.2966805, -13.8864756, 7.2316446, -21.3594875, 21.1831551
23: -12.7243576, 8.1692791, -12.6453457, 8.0867376, -19.2244339, 19.2367516
24: -18.1735649, 7.0968804, -18.0974979, 6.9688120, -25.1423759, 25.1943779
25: -10.9349003, 8.3927994, -10.8449860, 8.2996101, -19.2345104, 19.2377853
26: -14.3044472, 15.6817474, -14.1273136, 15.5860405, -29.8904877, 29.8090611
27: -24.8455200, 7.1606884, -24.7633095, 7.0663772, -31.9118977, 31.9239979
28: -14.9510117, 9.0561609, -14.8435163, 9.0027828, -23.2106628, 23.1649818
29: -18.4125557, 10.1790972, -18.1081772, 10.1330872, -26.9664726, 26.6979637
30: -14.8828487, 13.4043531, -14.7636986, 13.2942581, -27.4195175, 27.3796005
31: -17.5769768, 2.9609833, -17.4872589, 2.8908291, -20.4678059, 20.4482422
32: -13.4014053, 13.6583633, -13.2502012, 13.5903320, -26.9917374, 26.9041443
33: -32.1279373, 7.0747275, -32.0287857, 6.8202715, -37.0444641, 37.2166443
34: -25.7431412, 7.9496279, -25.6685829, 7.7958174, -31.0896606, 31.1688614
35: -20.0119705, 11.5732346, -19.9157276, 11.4530096, -30.6063843, 30.6599045
36: -20.0747070, 15.7376480, -19.9318066, 15.6764259, -33.0821915, 32.9997864
37: -28.6395760, 7.6931419, -28.4997120, 7.3877530, -33.3829346, 33.5671158
38: -25.1139488, 16.9687786, -24.9628067, 16.8029518, -41.5313721, 41.5461731
39: -36.4872742, 8.8410425, -36.3892937, 8.6333141, -43.9664001, 44.1125946
40: -31.1945038, 1.3201714, -31.1122723, 1.0423126, -27.6803207, 27.8936920
41: -18.2788887, 12.0428667, -18.2086143, 11.8993101, -26.3215256, 26.3882523
42: -11.4198790, 11.6209841, -11.3256464, 11.5034370, -22.9233170, 22.9466305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=193, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.7477054, upper bound: 17.0492606
time: 29.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9077440, upper bound: 17.0494071
time: 34.28 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -31.3838081, 1.6036625, -31.3611164, 1.5845780, -32.9683876, 32.9647789
1: -14.7718906, 4.6486568, -14.7477350, 4.6116295, -19.3835201, 19.3963928
2: -19.7406349, 1.6434655, -19.7348785, 1.6282268, -19.6827736, 19.6892319
3: -19.3849735, 1.1556921, -19.3761616, 1.1616700, -19.9882736, 19.9635010
4: -26.5602474, 0.1902361, -26.5524864, 0.1494575, -24.7836761, 24.8159637
5: -23.2506828, 1.9241419, -23.2418098, 1.9197824, -21.7208214, 21.7040253
6: -18.7548733, 6.9172401, -18.7227974, 6.9149933, -24.8920898, 24.8399582
7: -24.9344959, 0.8891015, -24.9108028, 0.8661723, -22.5535011, 22.5412140
8: -35.0530357, -2.1290822, -35.0414658, -2.1675162, -30.0189667, 30.0473709
9: -13.0981121, 11.7994537, -13.0861378, 11.7696476, -24.8677597, 24.8855915
10: -12.9213123, 18.0873642, -12.9042873, 18.0310612, -30.9523735, 30.9916515
11: -12.4006720, 14.5740089, -12.2950039, 14.5664997, -25.7982101, 25.6947098
12: -0.3857288, 32.8948364, -0.2874842, 32.8854599, -30.7562866, 30.6632843
13: -21.0238762, 19.0101128, -21.0003014, 19.0280495, -39.4228516, 39.3883362
14: -45.0896835, -0.8125987, -45.0414543, -0.7540479, -34.3601379, 34.2372818
15: -22.5710411, -0.3901000, -22.5543938, -0.4634581, -20.7662659, 20.8179245
16: -18.1570549, 7.2459483, -18.1041145, 7.2064552, -25.3635101, 25.3500633
17: -34.9072266, 10.7934971, -34.6671295, 10.8153934, -43.4743195, 43.2082214
18: -11.9693909, 14.3650932, -11.9676628, 14.3435650, -26.3129559, 26.3327560
19: -15.8255291, 1.9929366, -15.7869892, 1.9843149, -17.8098450, 17.7799263
20: -10.9510431, 5.0887942, -10.9232244, 5.0560646, -16.0071068, 16.0120182
21: -12.1521244, 7.5358033, -12.0839891, 7.5204659, -19.6725903, 19.6197929
22: -14.0675068, 7.2841444, -13.9598751, 7.2668486, -21.3343544, 21.2440186
23: -12.6947947, 8.1160345, -12.6737862, 8.1051712, -19.2137833, 19.2035408
24: -18.1459656, 7.0339642, -18.1324577, 7.0043221, -25.1502876, 25.1664219
25: -10.9068413, 8.3609486, -10.8793964, 8.3272896, -19.2341309, 19.2403450
26: -14.2309980, 15.5909538, -14.1878033, 15.5928917, -29.8238907, 29.7787571
27: -24.8047562, 7.0801792, -24.7892799, 7.0821309, -31.8868866, 31.8694592
28: -14.9254837, 9.0200758, -14.8780651, 9.0194893, -23.2031021, 23.1506500
29: -18.3408813, 10.1750832, -18.1940575, 10.1732597, -26.9256058, 26.7776299
30: -14.8409538, 13.3352318, -14.7924480, 13.3183022, -27.4021378, 27.3469543
31: -17.5437584, 2.9344864, -17.5182819, 2.9150276, -20.4587860, 20.4527683
32: -13.3654547, 13.6365519, -13.3103046, 13.6360474, -27.0015030, 26.9347038
33: -32.0565491, 6.9608521, -32.0721512, 6.8717098, -37.0350418, 37.1424561
34: -25.7075939, 7.8762312, -25.7048721, 7.8340364, -31.1036453, 31.1384048
35: -19.9664726, 11.5252285, -19.9485817, 11.4817734, -30.6065674, 30.6438751
36: -20.0143929, 15.7071571, -19.9780350, 15.6944075, -33.0399933, 33.0185242
37: -28.5309792, 7.4994345, -28.5503178, 7.4291730, -33.3309326, 33.4240875
38: -25.0507317, 16.9346523, -25.0227222, 16.8418961, -41.5054474, 41.5745392
39: -36.4304466, 8.7665310, -36.4361610, 8.6722069, -43.9710388, 44.0828705
40: -31.1116829, 1.1430926, -31.1361313, 1.0809813, -27.6418152, 27.7298927
41: -18.2058430, 11.9148703, -18.2243118, 11.9130888, -26.2650833, 26.2820511
42: -11.3606396, 11.5255976, -11.3573494, 11.5175266, -22.8781662, 22.8829460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=193, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8072060, upper bound: 17.0492608
time: 31.29 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9254497, upper bound: 17.0494070
time: 36.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -31.5273552, 1.6831827, -31.3612251, 1.5978050, -33.1251602, 33.0444069
1: -14.8497410, 4.6820731, -14.7482777, 4.6162491, -19.4659901, 19.4303513
2: -19.8135071, 1.6973016, -19.7349358, 1.6371622, -19.7696190, 19.7389297
3: -19.5106411, 1.2333291, -19.3765068, 1.1734498, -20.1273117, 20.0256996
4: -26.6637611, 0.2705727, -26.5532589, 0.1594572, -24.9003448, 24.8934021
5: -23.3824978, 2.0147290, -23.2422714, 1.9336107, -21.8682327, 21.7806511
6: -18.8051968, 7.0008860, -18.7258778, 6.9163704, -24.9440994, 24.9384422
7: -25.0532036, 0.9588485, -24.9116879, 0.8764520, -22.6903191, 22.6052437
8: -35.1449203, -2.0515323, -35.0418243, -2.1559849, -30.1190720, 30.1086044
9: -13.1565800, 11.8402538, -13.0870504, 11.7714176, -24.9279976, 24.9273033
10: -12.9951935, 18.1399040, -12.9047003, 18.0323963, -31.0275898, 31.0446053
11: -12.4666767, 14.6287422, -12.2953033, 14.5669250, -25.8729172, 25.7470169
12: -0.4751034, 32.9969101, -0.2927933, 32.8872528, -30.8476715, 30.7598877
13: -21.1724930, 19.0858765, -21.0021896, 19.0408382, -39.5891571, 39.4634323
14: -45.3135109, -0.7166586, -45.0414047, -0.7345724, -34.6081772, 34.3191223
15: -22.6201019, -0.3450031, -22.5548458, -0.4641399, -20.8176727, 20.8639755
16: -18.1999626, 7.2838316, -18.1021576, 7.2071953, -25.4071579, 25.3859901
17: -35.1543350, 10.8634348, -34.6673508, 10.8271236, -43.7380905, 43.2747726
18: -12.0171785, 14.4671803, -11.9698257, 14.3443890, -26.3615685, 26.4370060
19: -15.8584595, 2.0191970, -15.7877703, 1.9852810, -17.8437405, 17.8069668
20: -10.9974566, 5.1108093, -10.9235420, 5.0563507, -16.0538063, 16.0343513
21: -12.2027493, 7.5645714, -12.0855045, 7.5213308, -19.7240791, 19.6500759
22: -14.1322374, 7.3018894, -13.9606085, 7.2613606, -21.3935986, 21.2624969
23: -12.7272968, 8.1704712, -12.6751823, 8.1060247, -19.2479095, 19.2702675
24: -18.1809196, 7.0990562, -18.1350174, 7.0045371, -25.1854572, 25.2340736
25: -10.9388084, 8.3942986, -10.8806515, 8.3272905, -19.2660980, 19.2749500
26: -14.3088360, 15.6827545, -14.1964388, 15.5935240, -29.9023590, 29.8791924
27: -24.8505554, 7.1634049, -24.7939606, 7.0830531, -31.9336090, 31.9573650
28: -14.9535732, 9.0600891, -14.8793087, 9.0207071, -23.2313080, 23.2103882
29: -18.4165878, 10.1899652, -18.1941910, 10.1714840, -27.0091171, 26.7982941
30: -14.8850155, 13.4078312, -14.7955017, 13.3184719, -27.4461288, 27.4264145
31: -17.5806675, 2.9623175, -17.5190735, 2.9155979, -20.4962654, 20.4813919
32: -13.4043608, 13.6711702, -13.3077908, 13.6390438, -27.0434036, 26.9789619
33: -32.1394234, 7.0772424, -32.0827866, 6.8722820, -37.1148834, 37.2762146
34: -25.7520027, 7.9517097, -25.7086067, 7.8348465, -31.1505432, 31.2130508
35: -20.0170784, 11.5740614, -19.9511681, 11.4821854, -30.6549149, 30.6961136
36: -20.0787468, 15.7405396, -19.9815941, 15.6949739, -33.1029892, 33.0559616
37: -28.6531410, 7.6942945, -28.5686073, 7.4291430, -33.4429779, 33.6399689
38: -25.1251125, 16.9701347, -25.0261078, 16.8423843, -41.6051025, 41.6089554
39: -36.5002594, 8.8424320, -36.4415283, 8.6723433, -44.0364380, 44.1670761
40: -31.2025299, 1.3220515, -31.1503277, 1.0824776, -27.7335129, 27.9261208
41: -18.2830811, 12.0454178, -18.2368088, 11.9136524, -26.3402710, 26.4270973
42: -11.4234829, 11.6232452, -11.3620710, 11.5187778, -22.9422607, 22.9853172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=194, inp2_unstable=194, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1407

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1416

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.8488878, upper bound: 17.0492608
time: 33.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -16.9670725, upper bound: 17.0494071
time: 21.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 57.32 seconds
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.7477054, upper bound: 17.0492606
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.8661213, upper bound: 17.0494071
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.7477054, upper bound: 17.0492606
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.9077440, upper bound: 17.0494071
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.8072060, upper bound: 17.0492608
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.9254497, upper bound: 17.0494070
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.8488878, upper bound: 17.0492608
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 57.32
Output dim: 12, lower bound: -16.9670725, upper bound: 17.0494071

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 93.21 + 1069.88 = 1163.09 seconds
