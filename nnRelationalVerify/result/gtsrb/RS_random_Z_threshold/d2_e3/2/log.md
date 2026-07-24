## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 13.8015208638


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0493317, 31.0493317)
1: (-0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5305252, 21.5305252)
2: (-0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0709457, 21.0709419)
3: (-4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1444664, 20.1444626)
4: (-7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8377457, 25.8377419)
5: (-4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0121880, 24.0121918)
6: (-39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8676071, 29.8676071)
7: (-9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9351921, 25.9351959)
8: (-13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4519997, 29.4519920)
9: (-8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9809189, 27.9809151)
10: (-29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8678818, 39.8678741)
11: (-26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8585052, 29.8585129)
12: (-46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5951309, 32.5951309)
13: (-32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6680450, 38.6680374)
14: (-59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1684265, 57.1684265)
15: (-14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1610107, 28.1610107)
16: (-15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7779236, 29.7779236)
17: (-59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5589142, 51.5589142)
18: (-22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1982040, 33.1982040)
19: (-22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1389008, 22.1388931)
20: (-27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5451355, 22.5451355)
21: (-26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0777054, 29.0777092)
22: (-29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6146393, 27.6146393)
23: (-17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1740494, 24.1740532)
24: (-16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9444122, 25.9444199)
25: (-23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0914383, 28.0914383)
26: (-39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4904404, 33.4904404)
27: (-19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381)
28: (-21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4750061, 26.4750137)
29: (-24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7213745, 28.7213745)
30: (-30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1219025, 34.1218948)
31: (-23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.2022552, 25.2022552)
32: (-37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9130325, 29.9130325)
33: (-54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6694031, 44.6694031)
34: (-49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9868469, 35.9868393)
35: (-40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1470184, 36.1470261)
36: (-45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8584671, 39.8584595)
37: (-60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0494537, 45.0494537)
38: (-53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0107880, 49.0107880)
39: (-61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4642181, 48.4642181)
40: (-50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1585846, 39.1585770)
41: (-32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0868225, 38.0868225)
42: (-30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1964989, 23.1964951)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 51.18 = 53.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -13.8153362, upper bound: 13.8153362

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1565

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1447

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8070213, upper bound: 13.8070213
time: 33.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8070213, upper bound: 13.8070213
time: 34.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 68.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 68.51
Output dim: 1, lower bound: -13.8070213, upper bound: 13.8070213
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 68.51
Output dim: 1, lower bound: -13.8070213, upper bound: 13.8070213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0493317, 31.0493317
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5304947, 21.5305214
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0694504, 21.0700035
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1438522, 20.1436462
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8344040, 25.8359375
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0114937, 24.0112648
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8669357, 29.8653564
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9351807, 25.9351501
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4511757, 29.4516029
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9790802, 27.9774399
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8651810, 39.8622589
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8567505, 29.8546982
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5934067, 32.5920563
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6677246, 38.6676254
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1678772, 57.1678009
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1583176, 28.1599388
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7757721, 29.7733765
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5572968, 51.5592804
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1970139, 33.1975098
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1387405, 22.1386833
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5455589, 22.5445328
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0777893, 29.0769653
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6125183, 27.6142654
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1739960, 24.1735954
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9426270, 25.9431686
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0910263, 28.0911789
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4896622, 33.4898376
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4749146, 26.4749756
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7209473, 28.7214890
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1217270, 34.1209412
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.2016220, 25.2014923
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9118652, 29.9101410
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6680298, 44.6681213
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9865570, 35.9864273
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1454620, 36.1456223
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8584290, 39.8583908
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0483856, 45.0488052
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0105743, 49.0101776
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4638214, 48.4636002
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1583862, 39.1572952
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0867004, 38.0854340
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1949615, 23.1931076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1573

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977747, upper bound: 13.7956787
time: 26.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7956787, upper bound: 13.7977747
time: 27.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0493317, 31.0493279
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5305176, 21.5304947
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0700073, 21.0694504
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1436462, 20.1438484
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8359375, 25.8344040
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0112648, 24.0114937
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8653564, 29.8669434
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9351501, 25.9351807
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4516029, 29.4511833
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9774399, 27.9790802
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8622513, 39.8651733
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546982, 29.8567505
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5920563, 32.5934067
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6676331, 38.6677322
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1678009, 57.1678925
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1599350, 28.1583214
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7733765, 29.7757721
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5592804, 51.5573120
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1975098, 33.1970139
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1386871, 22.1387367
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5445290, 22.5455589
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0769653, 29.0777893
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6142654, 27.6125183
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1735992, 24.1739960
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9431686, 25.9426270
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0911789, 28.0910301
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4898453, 33.4896622
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4749756, 26.4749146
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214890, 28.7209473
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1209335, 34.1217117
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.2014847, 25.2016182
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9101410, 29.9118652
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6681213, 44.6680298
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9864197, 35.9865646
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1456146, 36.1454544
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8583984, 39.8584213
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0488129, 45.0483932
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0101776, 49.0105896
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4635925, 48.4638214
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1572876, 39.1583939
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0854492, 38.0867081
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1931076, 23.1949615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1743

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8041647, upper bound: 13.8063834
time: 53.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8063834, upper bound: 13.8041647
time: 30.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 86.18 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 86.18
Output dim: 1, lower bound: -13.7977747, upper bound: 13.7956787
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 86.18
Output dim: 1, lower bound: -13.7956787, upper bound: 13.7977747
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 86.18
Output dim: 1, lower bound: -13.8041647, upper bound: 13.8063834
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 86.18
Output dim: 1, lower bound: -13.8063834, upper bound: 13.8041647

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0442200, 31.0457687
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5281982, 21.5293999
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0647354, 21.0661621
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1358109, 20.1411095
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8343277, 25.8328362
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0099068, 24.0146523
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8571625, 29.8605194
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9324265, 25.9351730
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4501381, 29.4507713
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9753418, 27.9772453
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8599167, 39.8619537
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546600, 29.8569107
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5918427, 32.5897598
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6486359, 38.6548538
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1559753, 57.1508331
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1597519, 28.1580887
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7687073, 29.7719269
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5525436, 51.5473938
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1807709, 33.1727448
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1372604, 22.1364365
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5367203, 22.5342484
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0734329, 29.0744553
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6079330, 27.6032600
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1681519, 24.1663361
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9366302, 25.9330063
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0849838, 28.0825424
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4719391, 33.4636765
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4681931, 26.4641495
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214432, 28.7209053
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1168060, 34.1159210
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1987762, 25.1981277
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9078217, 29.9097290
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6570129, 44.6599960
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9864426, 35.9865341
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1351166, 36.1384888
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8519897, 39.8530350
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0488739, 45.0483093
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0095978, 49.0100250
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4486694, 48.4531708
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1575012, 39.1583710
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0795288, 38.0822296
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1888809, 23.1915550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1688

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 514

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8018890, upper bound: 13.8041106
time: 32.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8018890, upper bound: 13.8041106
time: 31.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0457687, 31.0442200
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5294266, 21.5281715
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0667191, 21.0641785
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1408997, 20.1360168
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8343735, 25.8327904
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0144234, 24.0101318
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8589401, 29.8587418
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9351501, 25.9324532
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4511909, 29.4497147
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9756088, 27.9769783
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8590317, 39.8628387
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8548584, 29.8567162
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5884094, 32.5931931
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6547394, 38.6487350
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1507416, 57.1560516
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1597061, 28.1581345
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7695312, 29.7711029
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5493698, 51.5505676
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1732483, 33.1802750
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1363754, 22.1373062
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5332184, 22.5377502
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0736313, 29.0742569
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6050034, 27.6061821
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1659393, 24.1685486
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9335480, 25.9360886
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0826950, 28.0848389
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4638519, 33.4717712
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4642029, 26.4681320
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214432, 28.7209091
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1151581, 34.1175690
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1979980, 25.1989059
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9079971, 29.9095459
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6600952, 44.6569290
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9863968, 35.9865799
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1386566, 36.1349411
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8529968, 39.8520279
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0487213, 45.0484619
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0096283, 49.0100098
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4529419, 48.4488907
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1572723, 39.1585922
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0809631, 38.0807953
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1897049, 23.1907349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8063080, upper bound: 13.8026916
time: 38.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8049073, upper bound: 13.8040892
time: 27.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 68.25 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.25
Output dim: 1, lower bound: -13.8018890, upper bound: 13.8041106
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.25
Output dim: 1, lower bound: -13.8018890, upper bound: 13.8041106
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.25
Output dim: 1, lower bound: -13.8063080, upper bound: 13.8026916
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.25
Output dim: 1, lower bound: -13.8049073, upper bound: 13.8040892

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0442047, 31.0457420
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5281677, 21.5293350
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0646591, 21.0660439
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1357422, 20.1412125
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8342972, 25.8327255
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0099068, 24.0146408
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8571625, 29.8605118
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9324036, 25.9351578
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4500771, 29.4509010
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9754410, 27.9771729
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8600769, 39.8617096
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546677, 29.8569489
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5916290, 32.5897141
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6486588, 38.6548615
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1556244, 57.1507416
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1599579, 28.1580544
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7688904, 29.7717972
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5523758, 51.5473785
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1806183, 33.1727295
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1372452, 22.1364861
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5367203, 22.5344009
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0733795, 29.0747604
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6079102, 27.6032562
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1681061, 24.1664925
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9366074, 25.9330673
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0849838, 28.0827179
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4719391, 33.4636841
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4681854, 26.4641876
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214508, 28.7208862
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1167831, 34.1158981
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1987305, 25.1983147
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9078751, 29.9097214
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6568909, 44.6600952
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9866333, 35.9865265
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1350403, 36.1385040
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8519897, 39.8530273
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0487823, 45.0483017
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0095673, 49.0100861
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4486389, 48.4531631
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1576843, 39.1583252
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0796356, 38.0822220
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1888657, 23.1917191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 942

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 782

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8017959, upper bound: 13.8001046
time: 26.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7978861, upper bound: 13.8040176
time: 26.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0442200, 31.0457535
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5281982, 21.5293732
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0647354, 21.0660896
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1358109, 20.1410408
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8343277, 25.8328094
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0099068, 24.0146523
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8571625, 29.8605194
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9324265, 25.9351578
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4501381, 29.4507065
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9752655, 27.9772453
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8596649, 39.8619537
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546600, 29.8569145
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5917969, 32.5897598
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6486435, 38.6548538
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1558838, 57.1508331
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1597137, 28.1580887
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7685776, 29.7719269
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5525284, 51.5473938
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1807480, 33.1727448
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1372604, 22.1364174
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5367203, 22.5342445
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0734329, 29.0744019
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6079254, 27.6032600
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1681519, 24.1662941
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9366302, 25.9329834
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0849838, 28.0825386
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4719543, 33.4636765
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4681931, 26.4641342
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214279, 28.7209053
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1167831, 34.1159210
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1987762, 25.1980743
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9078140, 29.9097290
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6570129, 44.6598740
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9864349, 35.9865341
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1351166, 36.1384125
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8519897, 39.8530350
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0488739, 45.0483093
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0095978, 49.0099945
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4486694, 48.4531403
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1574554, 39.1583710
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0795135, 38.0822296
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1888809, 23.1915321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1538

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8017184, upper bound: 13.8009641
time: 31.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7987429, upper bound: 13.8039394
time: 35.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0454941, 31.0439148
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5294037, 21.5281448
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0669937, 21.0644417
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1409264, 20.1360359
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8336487, 25.8320236
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0125122, 24.0081329
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8580246, 29.8574142
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9345512, 25.9318390
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4508972, 29.4493942
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9734116, 27.9745178
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8521652, 39.8551178
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546219, 29.8565407
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5857010, 32.5901184
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6550903, 38.6488571
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1481476, 57.1529999
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1584702, 28.1567650
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7633514, 29.7641678
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5485840, 51.5496216
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1719971, 33.1791534
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1333618, 22.1344566
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5295334, 22.5343933
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0695343, 29.0707169
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6035767, 27.6051788
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1610260, 24.1640892
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9293060, 25.9322968
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0844498, 28.0871201
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4587250, 33.4669647
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4590225, 26.4634476
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214127, 28.7210083
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1162872, 34.1187973
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1953659, 25.1965675
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9103394, 29.9115906
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6566315, 44.6538162
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9888840, 35.9886856
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1381226, 36.1344223
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8546448, 39.8535690
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0492096, 45.0489807
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0090637, 49.0095444
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4547577, 48.4505310
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1595612, 39.1604156
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0834656, 38.0829468
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1893044, 23.1903725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1403

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 521

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8061968, upper bound: 13.7954796
time: 34.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7990954, upper bound: 13.8025822
time: 32.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0454712, 31.0439377
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5294037, 21.5281448
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0669785, 21.0644569
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1409187, 20.1360435
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8336029, 25.8320732
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0124207, 24.0082245
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8576050, 29.8578339
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9345284, 25.9318542
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4508820, 29.4494171
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9731445, 27.9747810
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8513107, 39.8559647
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546829, 29.8564796
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5853348, 32.5904846
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6548615, 38.6490860
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1476746, 57.1534576
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1583405, 28.1568985
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7626038, 29.7649231
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5484314, 51.5497742
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1721191, 33.1790314
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1335297, 22.1342812
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5298691, 22.5340576
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0700912, 29.0701561
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6040039, 27.6047516
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1614761, 24.1636353
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9297485, 25.9318390
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0849686, 28.0866013
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4590454, 33.4666443
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4595108, 26.4629593
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7215424, 28.7208786
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1163788, 34.1187057
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1956558, 25.1962738
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9100494, 29.9118805
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6569977, 44.6534653
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9885025, 35.9890518
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1381226, 36.1343994
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8545380, 39.8536682
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0492401, 45.0489578
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0091553, 49.0094604
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4545898, 48.4506989
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1590881, 39.1608810
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0831146, 38.0833054
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1893501, 23.1903305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1588

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1665

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8040377, upper bound: 13.7821955
time: 31.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7830070, upper bound: 13.8032245
time: 27.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 61.70 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.8017959, upper bound: 13.8001046
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.7978861, upper bound: 13.8040176
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.8017184, upper bound: 13.8009641
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.7987429, upper bound: 13.8039394
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.8061968, upper bound: 13.7954796
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.7990954, upper bound: 13.8025822
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.8040377, upper bound: 13.7821955
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.70
Output dim: 1, lower bound: -13.7830070, upper bound: 13.8032245

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0442734, 31.0458031
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5269508, 21.5280304
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0662155, 21.0674858
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1368332, 20.1422462
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8301926, 25.8281288
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0054741, 24.0096359
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8550568, 29.8572769
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9301529, 25.9326706
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4490051, 29.4496574
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9727936, 27.9741440
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8508224, 39.8507462
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8538666, 29.8566132
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5893097, 32.5870819
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6475067, 38.6530838
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1531830, 57.1479340
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1574478, 28.1551971
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7590408, 29.7602005
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5498352, 51.5443420
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1793823, 33.1721039
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1307831, 22.1308594
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5303078, 22.5293045
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0684891, 29.0716019
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6057510, 27.6020279
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1570816, 24.1570435
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9293137, 25.9268112
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0885239, 28.0875969
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4586487, 33.4522552
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4580536, 26.4553604
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7212372, 28.7210884
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1199646, 34.1194534
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1941986, 25.1944122
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9125977, 29.9138412
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6535339, 44.6571503
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9935455, 35.9922867
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1351395, 36.1386108
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8550110, 39.8558197
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0485077, 45.0480499
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0074463, 49.0083237
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4538116, 48.4578094
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1624451, 39.1616364
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0844879, 38.0862503
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1863251, 23.1894379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 973

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.8008785, upper bound: 13.7821043
time: 33.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7837883, upper bound: 13.7991757
time: 29.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0442657, 31.0458145
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5268593, 21.5281143
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0661011, 21.0675964
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1367722, 20.1423035
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8297043, 25.8286171
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0049019, 24.0102081
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8539276, 29.8584061
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9299240, 25.9328995
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4488373, 29.4498405
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9724197, 27.9745216
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8491135, 39.8524551
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8543320, 29.8561478
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5889969, 32.5873985
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6468811, 38.6537018
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1528015, 57.1483002
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1570969, 28.1555519
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7572937, 29.7619476
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5493469, 51.5448456
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1799927, 33.1714935
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1316147, 22.1300316
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5316200, 22.5279961
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0702133, 29.0698700
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6066818, 27.6010971
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1586609, 24.1554680
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9303436, 25.9257736
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0898590, 28.0862617
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4605255, 33.4503784
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4593506, 26.4540634
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7216568, 28.7206726
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1203308, 34.1190796
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1948242, 25.1937790
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9119949, 29.9144440
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6539459, 44.6567383
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9923859, 35.9934540
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1351395, 36.1386032
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8547821, 39.8560486
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0485229, 45.0480423
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0078125, 49.0079727
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4532928, 48.4583282
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1609955, 39.1630783
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0836792, 38.0870514
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1865692, 23.1891823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1003

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7963594, upper bound: 13.8024908
time: 27.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7963594, upper bound: 13.8024908
time: 27.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0438232, 31.0451698
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5280914, 21.5292282
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0645714, 21.0659180
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1357155, 20.1409607
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8339462, 25.8327255
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0097618, 24.0145340
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8564606, 29.8593369
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9322701, 25.9349823
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4496765, 29.4501114
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9734879, 27.9761543
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8558273, 39.8596115
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546524, 29.8569031
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5912933, 32.5892334
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6485214, 38.6548080
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1557312, 57.1506500
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1587143, 28.1573753
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7665863, 29.7707253
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5525970, 51.5471191
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1802597, 33.1719437
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1370773, 22.1361885
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5370255, 22.5340614
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0736160, 29.0743027
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6078796, 27.6031418
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1674347, 24.1651878
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9364624, 25.9324875
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0855942, 28.0824356
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4717865, 33.4631577
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4680939, 26.4636612
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7214127, 28.7208710
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1167526, 34.1158905
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1986694, 25.1976814
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9077988, 29.9098053
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6565247, 44.6592712
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9862061, 35.9867859
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1349640, 36.1382065
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8519135, 39.8529282
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0488586, 45.0482712
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0083618, 49.0080490
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4486237, 48.4530563
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1574554, 39.1583481
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0794983, 38.0823364
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1875381, 23.1893959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 955

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1431

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7917419, upper bound: 13.7959348
time: 26.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7966834, upper bound: 13.7909914
time: 34.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0436325, 31.0453568
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5280533, 21.5292664
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0645638, 21.0659332
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1357307, 20.1409492
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8342438, 25.8324280
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0097923, 24.0145073
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8559647, 29.8598328
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9322472, 25.9350052
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4495468, 29.4502487
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9741745, 27.9754639
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8573380, 39.8581085
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8546524, 29.8569031
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5912704, 32.5892601
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6485825, 38.6547241
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1556854, 57.1506958
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1589966, 28.1570892
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7673721, 29.7699356
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5522461, 51.5474548
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1799622, 33.1722412
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1370163, 22.1362419
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5365372, 22.5345497
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0733261, 29.0745850
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6078033, 27.6032219
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1670532, 24.1655769
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9361267, 25.9328232
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0848846, 28.0831490
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4714355, 33.4635162
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4677124, 26.4640427
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7213898, 28.7208939
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1167526, 34.1158829
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1983795, 25.1979637
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9078903, 29.9097137
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6564178, 44.6593781
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9866943, 35.9862976
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1349030, 36.1382675
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8518829, 39.8529510
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0488281, 45.0482864
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0076599, 49.0087509
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4485931, 48.4530869
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1574554, 39.1583481
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0796204, 38.0822220
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1867447, 23.1901779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1606

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7932581, upper bound: 13.8033422
time: 28.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7983374, upper bound: 13.7989066
time: 34.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0459213, 31.0444183
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5302734, 21.5288582
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0679398, 21.0651855
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1482201, 20.1436539
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8312569, 25.8291550
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0161362, 24.0122337
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8516006, 29.8520584
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9371376, 25.9339676
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4539413, 29.4525871
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9722366, 27.9733582
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8536148, 39.8564835
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8568573, 29.8584061
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5905914, 32.5951462
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6577148, 38.6511002
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1376648, 57.1410065
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1582794, 28.1566391
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7647858, 29.7654877
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5386658, 51.5386658
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1649933, 33.1705322
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1319542, 22.1325951
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5301666, 22.5351295
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0700455, 29.0711517
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6001892, 27.6019135
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1609116, 24.1640358
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9271088, 25.9296646
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0800781, 28.0829773
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4568405, 33.4648361
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4591370, 26.4635735
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7228394, 28.7227631
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1194611, 34.1224442
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1902466, 25.1905518
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9087067, 29.9101639
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6529846, 44.6507721
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9866180, 35.9868164
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1369934, 36.1333237
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8584061, 39.8571930
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0517883, 45.0521011
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0105591, 49.0105743
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4616089, 48.4569244
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1572113, 39.1584625
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0795898, 38.0797119
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1802139, 23.1827965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1679

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1451

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7937142, upper bound: 13.7818605
time: 31.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7925285, upper bound: 13.7830488
time: 32.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0459976, 31.0443382
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5301208, 21.5290108
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0677414, 21.0653915
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1485481, 20.1433296
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8307762, 25.8296318
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0166168, 24.0117569
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8526688, 29.8509903
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9366798, 25.9344292
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4540939, 29.4524345
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9722519, 27.9733505
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8535233, 39.8565750
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8564835, 29.8587761
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5907288, 32.5950089
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6573181, 38.6514816
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1361694, 57.1425018
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1583481, 28.1565742
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7646713, 29.7656021
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5376282, 51.5396729
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1633835, 33.1721420
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1314964, 22.1330528
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5302658, 22.5350342
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0699692, 29.0712280
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6003113, 27.6017914
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1609726, 24.1639748
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9266739, 25.9300995
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0803070, 28.0827484
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4565964, 33.4650803
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4591522, 26.4635582
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7231750, 28.7224350
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1199341, 34.1219635
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1893539, 25.1914406
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9089127, 29.9099579
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6535797, 44.6501694
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9870148, 35.9864426
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1370239, 36.1332932
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8582687, 39.8573380
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0523376, 45.0515442
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0101013, 49.0110321
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4611511, 48.4573822
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1576080, 39.1580658
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0802307, 38.0790634
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1817245, 23.1812859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1658

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1582

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7800021, upper bound: 13.8020580
time: 32.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7985713, upper bound: 13.7835289
time: 28.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0376053, 31.0345535
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5044785, 21.4987869
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0478287, 21.0420876
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1090775, 20.0990372
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8102493, 25.8046837
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9928932, 23.9853592
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8537369, 29.8538284
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9121475, 25.9054260
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4302368, 29.4239769
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9603043, 27.9603157
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8350143, 39.8391571
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8406067, 29.8442154
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5805130, 32.5888672
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6445007, 38.6456146
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1414490, 57.1487579
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1473312, 28.1440506
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7397766, 29.7390366
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5132294, 51.5197449
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1674271, 33.1727219
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0941238, 22.1007156
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5040283, 22.5119781
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0414810, 29.0461845
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5743484, 27.5803375
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1346741, 24.1406784
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9123077, 25.9169312
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0402222, 28.0485268
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4387207, 33.4483414
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4383621, 26.4448586
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7042542, 28.7081528
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1076508, 34.1113815
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1588516, 25.1648521
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9082336, 29.9102249
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6281128, 44.6310120
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9872131, 35.9852448
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1209793, 36.1196671
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8328476, 39.8376083
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0178833, 45.0256500
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9942627, 48.9999847
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4178925, 48.4232178
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1539612, 39.1551971
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0799789, 38.0799561
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1858177, 23.1870918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1583

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7968269, upper bound: 13.7819488
time: 27.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8037917, upper bound: 13.7749906
time: 26.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0360794, 31.0360832
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5000458, 21.5032196
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0446167, 21.0453033
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1039124, 20.1041985
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8062210, 25.8087120
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9895592, 23.9886971
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8535995, 29.8539734
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9081039, 25.9094696
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4254303, 29.4287758
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9586792, 27.9619370
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8345108, 39.8396759
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8424072, 29.8424034
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5837250, 32.5856552
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6513824, 38.6387253
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1429749, 57.1472321
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1454849, 28.1458893
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7367096, 29.7420959
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5183868, 51.5145874
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1658096, 33.1743317
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0999680, 22.0948792
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5077820, 22.5082207
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0461197, 29.0415421
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5795822, 27.5751038
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1385193, 24.1368332
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9148407, 25.9143982
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0468979, 28.0418472
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4407349, 33.4463272
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4414139, 26.4418068
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7088165, 28.7035904
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1090546, 34.1099854
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1642380, 25.1594734
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9083939, 29.9100723
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6345520, 44.6245804
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9846954, 35.9877625
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1234055, 36.1172485
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8384781, 39.8319778
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0259399, 45.0176086
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9996643, 48.9945679
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4271088, 48.4139938
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1534119, 39.1557465
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0797653, 38.0801773
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1861076, 23.1868019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1743

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1419

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7824491, upper bound: 13.8026493
time: 28.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7824320, upper bound: 13.8026665
time: 31.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 61.64 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.8008785, upper bound: 13.7821043
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7837883, upper bound: 13.7991757
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7963594, upper bound: 13.8024908
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7963594, upper bound: 13.8024908
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7917419, upper bound: 13.7959348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7966834, upper bound: 13.7909914
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7932581, upper bound: 13.8033422
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7983374, upper bound: 13.7989066
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7937142, upper bound: 13.7818605
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7925285, upper bound: 13.7830488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7800021, upper bound: 13.8020580
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7985713, upper bound: 13.7835289
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7968269, upper bound: 13.7819488
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.8037917, upper bound: 13.7749906
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7824491, upper bound: 13.8026493
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 61.64
Output dim: 1, lower bound: -13.7824320, upper bound: 13.8026665

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0430336, 31.0452042
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5248108, 21.5260620
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0639610, 21.0654716
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1362610, 20.1417923
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8269043, 25.8258591
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0043831, 24.0097275
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8493958, 29.8537674
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9289551, 25.9320068
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4501724, 29.4510689
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9714584, 27.9743195
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8483582, 39.8518524
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8469696, 29.8484192
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5844879, 32.5822296
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6433563, 38.6503525
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1519470, 57.1468048
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1494522, 28.1480942
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7585449, 29.7637215
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5488434, 51.5442200
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1748428, 33.1655884
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1312027, 22.1297073
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5289078, 22.5244370
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0642242, 29.0637474
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6067047, 27.6011124
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1550064, 24.1518288
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9311218, 25.9266586
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0909195, 28.0874214
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4492493, 33.4372635
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4568558, 26.4516830
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7226105, 28.7219429
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1151886, 34.1138992
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1948624, 25.1941109
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9084778, 29.9103928
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6533661, 44.6561737
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9917145, 35.9926682
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1343918, 36.1378174
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8547745, 39.8559113
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0472260, 45.0467606
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0076599, 49.0067825
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4512634, 48.4563293
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1614075, 39.1632767
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0836945, 38.0870972
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1796951, 23.1821899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 942

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7810533, upper bound: 13.7869139
time: 28.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7807846, upper bound: 13.7871826
time: 23.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0436592, 31.0445747
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5248108, 21.5260544
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0639763, 21.0654640
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1362610, 20.1417923
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8269424, 25.8258209
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0044212, 24.0096893
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8492889, 29.8538742
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9290237, 25.9319420
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4500656, 29.4511833
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9722137, 27.9735641
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8485107, 39.8516998
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8466034, 29.8487854
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5838242, 32.5828934
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6435242, 38.6501846
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1512909, 57.1474609
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1496353, 28.1479111
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7590714, 29.7631912
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5487213, 51.5443420
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1740875, 33.1663437
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1312866, 22.1296158
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5280609, 22.5252800
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0640945, 29.0638733
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6067047, 27.6011124
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1550217, 24.1518135
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9312286, 25.9265518
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0910187, 28.0873184
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4474030, 33.4391022
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4569778, 26.4515610
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7229233, 28.7216263
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1151581, 34.1139374
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1951599, 25.1938133
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9079437, 29.9109268
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6533966, 44.6561508
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9915924, 35.9927902
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1343613, 36.1378632
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8546524, 39.8560333
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0472412, 45.0467377
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0066223, 49.0078201
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4512939, 48.4562988
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1611938, 39.1634903
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0837250, 38.0870819
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1795883, 23.1823044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1583

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7870161, upper bound: 13.8017035
time: 32.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7955723, upper bound: 13.7931477
time: 32.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0366669, 31.0370560
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5374298, 21.5377426
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0642052, 21.0642471
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1682854, 20.1675949
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8371735, 25.8352547
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0345039, 24.0355225
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8521118, 29.8552780
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9510574, 25.9514351
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4519234, 29.4526100
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9677505, 27.9682961
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8483734, 39.8511810
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8585739, 29.8605537
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5825348, 32.5840797
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6179733, 38.6190414
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1163177, 57.1177368
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1571732, 28.1553116
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7676773, 29.7703056
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5197678, 51.5200958
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1354218, 33.1352081
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1427078, 22.1425362
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5246964, 22.5243568
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0769043, 29.0777512
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5981903, 27.5944633
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1723862, 24.1722908
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9260330, 25.9244003
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0746765, 28.0737305
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4625092, 33.4615860
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4806442, 26.4797020
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7182617, 28.7174683
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1102753, 34.1100845
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1961136, 25.1961365
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9025345, 29.9035492
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6316071, 44.6296158
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9832764, 35.9827957
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1135406, 36.1126785
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8420410, 39.8422241
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0607605, 45.0612869
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0017242, 49.0025406
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4200287, 48.4203873
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1707306, 39.1738510
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0753326, 38.0775604
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1819229, 23.1846390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1450

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7848585, upper bound: 13.7949156
time: 32.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7848585, upper bound: 13.7949156
time: 31.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0439072, 31.0427017
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5181923, 21.5186958
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0558777, 21.0557747
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1417046, 20.1372757
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8095131, 25.8124313
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -24.0070152, 24.0032578
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8426743, 29.8395767
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9290009, 25.9277344
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4358749, 29.4371796
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9662704, 27.9700661
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8200455, 39.8286285
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8744812, 29.8736496
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5934219, 32.5941124
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6248550, 38.6129761
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1463470, 57.1516724
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1399384, 28.1411362
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7324600, 29.7389336
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5507355, 51.5514832
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1564865, 33.1660385
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1202621, 22.1199455
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5112724, 22.5121422
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0639572, 29.0608215
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5990829, 27.5988541
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1599579, 24.1601562
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9276657, 25.9303284
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0927734, 28.0907249
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4522552, 33.4580536
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4522705, 26.4559898
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7313919, 28.7293854
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1235809, 34.1251907
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1853180, 25.1850128
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8987732, 29.8981705
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6259003, 44.6172867
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9932022, 35.9935760
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1194000, 36.1123505
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8263245, 39.8193741
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0292053, 45.0238342
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9533386, 48.9439011
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4238129, 48.4133987
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1648560, 39.1665268
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0796051, 38.0784454
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1646423, 23.1605873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1587

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7782281, upper bound: 13.7995135
time: 32.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7774617, upper bound: 13.8002851
time: 32.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0382118, 31.0351944
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5025597, 21.4967728
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0439987, 21.0371475
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1039276, 20.0927696
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7985458, 25.7904015
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9875488, 23.9789810
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8466797, 29.8478088
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9099274, 25.9029808
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4234772, 29.4162903
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9582596, 27.9576988
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8269882, 39.8299866
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8470917, 29.8531342
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5787048, 32.5890579
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6313248, 38.6340332
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1454163, 57.1534729
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1350250, 28.1294632
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7372513, 29.7360954
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5159454, 51.5229950
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1659012, 33.1711273
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0867538, 22.0943947
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4959488, 22.5049858
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0338593, 29.0406952
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5742798, 27.5804367
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1304626, 24.1381416
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9118576, 25.9167099
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0414734, 28.0514450
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4372406, 33.4474564
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4337845, 26.4408493
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7055435, 28.7097511
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1086273, 34.1126938
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1544113, 25.1614456
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9050369, 29.9073257
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6205902, 44.6245346
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9930115, 35.9900436
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1152649, 36.1147614
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8192825, 39.8257751
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0012054, 45.0116806
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9676971, 48.9769821
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4034576, 48.4106064
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1550903, 39.1561356
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0760498, 38.0759964
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1756516, 23.1791306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1450

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 985

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7858411, upper bound: 13.7743017
time: 31.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8031048, upper bound: 13.7570278
time: 27.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0362434, 31.0362968
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4989128, 21.5021706
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0440788, 21.0445366
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1023598, 20.1029396
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8055153, 25.8080139
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9895058, 23.9886627
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8531036, 29.8535690
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9006424, 25.9013481
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4257393, 29.4287643
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9550552, 27.9587250
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8242340, 39.8289719
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8231354, 29.8218765
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5836945, 32.5860748
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6533203, 38.6414032
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1446686, 57.1482086
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1325912, 28.1336174
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7258148, 29.7297516
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5187225, 51.5147705
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1614609, 33.1690063
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0930176, 22.0867767
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5056305, 22.5058403
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0380707, 29.0322647
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5730133, 27.5692825
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1359406, 24.1336899
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9147644, 25.9142303
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0487976, 28.0435753
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4373703, 33.4422531
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4407501, 26.4409637
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7088089, 28.7034225
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1050644, 34.1058273
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1568527, 25.1511650
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9055405, 29.9076920
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6197357, 44.6113815
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9700241, 35.9748688
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1122742, 36.1069717
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8295441, 39.8237610
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0205841, 45.0123825
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9900665, 48.9860916
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4247437, 48.4118576
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1525574, 39.1549683
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0808105, 38.0814362
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1860085, 23.1867294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7816503, upper bound: 13.8024094
time: 30.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7822111, upper bound: 13.8018491
time: 29.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0362968, 31.0362434
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4989891, 21.5020828
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0438499, 21.0447731
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1026573, 20.1026382
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8055153, 25.8080139
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9895287, 23.9886475
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8531952, 29.8534775
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8999786, 25.9020081
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4254265, 29.4290810
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9554749, 27.9583054
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8238068, 39.8293991
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8218842, 29.8231277
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5841446, 32.5856247
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6540680, 38.6406555
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1439667, 57.1489258
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1332169, 28.1329994
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7243652, 29.7311935
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5185699, 51.5149231
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1604767, 33.1699829
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0918655, 22.0879288
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5054092, 22.5060654
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0368423, 29.0334930
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5737686, 27.5685310
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1353760, 24.1342583
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9146652, 25.9143219
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0486221, 28.0437469
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4366684, 33.4429474
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4405823, 26.4411316
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7086487, 28.7035866
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1048813, 34.1060104
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1559296, 25.1520882
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9060059, 29.9072189
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6213379, 44.6097717
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9717941, 35.9730835
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1131287, 36.1061172
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8302765, 39.8230362
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0207062, 45.0122681
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9911957, 48.9849777
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4249725, 48.4116287
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1526184, 39.1549072
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0810242, 38.0812378
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1860237, 23.1867104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1304

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7772983, upper bound: 13.8020971
time: 39.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7818690, upper bound: 13.7975279
time: 24.14 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 65.64 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7810533, upper bound: 13.7869139
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7807846, upper bound: 13.7871826
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7870161, upper bound: 13.8017035
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7955723, upper bound: 13.7931477
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7848585, upper bound: 13.7949156
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7848585, upper bound: 13.7949156
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7782281, upper bound: 13.7995135
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7774617, upper bound: 13.8002851
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7858411, upper bound: 13.7743017
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.8031048, upper bound: 13.7570278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7816503, upper bound: 13.8024094
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7822111, upper bound: 13.8018491
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7772983, upper bound: 13.8020971
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.64
Output dim: 1, lower bound: -13.7818690, upper bound: 13.7975279

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0316086, 31.0338707
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5159149, 21.5180130
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0511894, 21.0548897
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1356010, 20.1411972
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8222198, 25.8234940
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9954643, 24.0021744
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8302307, 29.8306351
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9213791, 25.9258270
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4446411, 29.4473228
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9591751, 27.9630737
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8344116, 39.8397751
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8465958, 29.8487511
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5994949, 32.5946884
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6486969, 38.6550446
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1510468, 57.1472168
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1483841, 28.1468048
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7463226, 29.7551880
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5616608, 51.5549011
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1798553, 33.1726990
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1356125, 22.1340179
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5203285, 22.5167656
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0663681, 29.0662231
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6051865, 27.5984421
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1560822, 24.1527481
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9307327, 25.9259644
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0883713, 28.0835876
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4437943, 33.4355774
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4524384, 26.4463158
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7250900, 28.7225189
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1096039, 34.1058655
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1975250, 25.1959534
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8937225, 29.8935699
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6277771, 44.6246567
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9633408, 35.9584885
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1192093, 36.1193848
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8410721, 39.8394699
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0330200, 45.0297089
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9881439, 48.9855042
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4398041, 48.4435730
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1456757, 39.1444244
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0763474, 38.0787048
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1668701, 23.1656647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1497

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7862399, upper bound: 13.7918228
time: 31.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7771369, upper bound: 13.8009270
time: 29.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0369682, 31.0329208
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.5002518, 21.4934425
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0422363, 21.0342216
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0977669, 20.0842514
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7972488, 25.7876854
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9846115, 23.9740601
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8466492, 29.8477936
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9075470, 25.8997612
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4228821, 29.4155884
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9575958, 27.9568863
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8267670, 39.8336716
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8448257, 29.8511276
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5754318, 32.5870819
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6241074, 38.6212387
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1452637, 57.1546783
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1330643, 28.1274567
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7325363, 29.7314224
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5155487, 51.5226746
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1575317, 33.1638489
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0821304, 22.0917015
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4906311, 22.5027275
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0300598, 29.0386658
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5728149, 27.5789948
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1245193, 24.1347923
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9090271, 25.9145355
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0399933, 28.0509644
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4278412, 33.4412918
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4279861, 26.4377747
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7049942, 28.7093430
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1053696, 34.1112442
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1515274, 25.1597443
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9048767, 29.9072418
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6131134, 44.6193390
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9915924, 35.9899521
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1142273, 36.1145248
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8192673, 39.8255997
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9998322, 45.0107880
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9686584, 48.9768143
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4024658, 48.4055862
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1516266, 39.1534500
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0758667, 38.0758209
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1734543, 23.1776810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1348

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7887030, upper bound: 13.7500140
time: 31.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7960891, upper bound: 13.7426206
time: 33.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0319672, 31.0322151
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4984818, 21.5017357
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0412216, 21.0419235
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0991745, 20.1005783
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8057556, 25.8083420
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9905090, 23.9907684
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8529739, 29.8534470
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9003029, 25.9010239
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4244385, 29.4273071
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9534149, 27.9570465
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8257141, 39.8300781
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8178253, 29.8160248
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5808945, 32.5833511
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6286850, 38.6197281
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1446838, 57.1482697
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1342163, 28.1364746
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7280350, 29.7316895
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5188370, 51.5148926
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1529312, 33.1589966
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0854912, 22.0783234
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4963913, 22.4951706
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0301285, 29.0233727
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5708694, 27.5671158
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1274796, 24.1239777
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9067154, 25.9048157
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0406418, 28.0344429
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4302673, 33.4337845
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4290543, 26.4274139
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7102127, 28.7049332
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0956268, 34.0947571
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1472778, 25.1408005
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9040070, 29.9062729
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6183472, 44.6101532
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9701691, 35.9750290
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1095657, 36.1053925
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8231201, 39.8191910
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0173645, 45.0093613
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9881592, 48.9855194
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4131470, 48.4017487
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1554260, 39.1571808
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0789948, 38.0797882
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1899185, 23.1902618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7814058, upper bound: 13.7970088
time: 28.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769874, upper bound: 13.8022649
time: 31.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0321579, 31.0320282
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4984741, 21.5017357
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0414734, 21.0416718
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0999985, 20.0997581
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8058472, 25.8082542
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9916077, 23.9896698
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8529816, 29.8534317
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9003181, 25.9010162
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4242859, 29.4274635
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9533691, 27.9570847
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8253326, 39.8304443
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8172836, 29.8165627
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5809708, 32.5832748
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6316452, 38.6167679
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1447144, 57.1482391
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1354523, 28.1352386
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7277527, 29.7319717
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5188522, 51.5148773
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1514511, 33.1604767
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0845604, 22.0792465
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4949570, 22.4966049
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0291672, 29.0243340
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5708466, 27.5671387
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1262283, 24.1252251
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9053497, 25.9061813
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0396652, 28.0354233
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4288940, 33.4351425
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4271927, 26.4292755
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7103195, 28.7048187
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0940094, 34.0963745
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1464996, 25.1415863
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9041214, 29.9061508
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6184998, 44.6100006
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9701691, 35.9750290
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1106949, 36.1042633
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8249664, 39.8173447
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0175781, 45.0091553
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9895020, 48.9841614
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4146423, 48.4002533
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1547852, 39.1578369
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0791779, 38.0796127
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1895447, 23.1906281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1716

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7821263, upper bound: 13.7988682
time: 29.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7792311, upper bound: 13.8017644
time: 32.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0362778, 31.0362396
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4953766, 21.4989662
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0405807, 21.0418892
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0964432, 20.0972023
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8024712, 25.8053665
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9858208, 23.9853668
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8531952, 29.8534698
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8911057, 25.8942757
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4254227, 29.4290390
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9533768, 27.9567032
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8214874, 39.8275070
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8155670, 29.8178673
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5835953, 32.5847435
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6404724, 38.6296921
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1424561, 57.1472626
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1276016, 28.1268158
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7168961, 29.7253418
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5178146, 51.5130157
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1533432, 33.1613617
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0918694, 22.0879288
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5028687, 22.5028763
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0367661, 29.0333023
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5670242, 27.5603523
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1321182, 24.1303406
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9119568, 25.9110031
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0451660, 28.0394135
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4271545, 33.4310684
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4337082, 26.4330254
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7041168, 28.6983109
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1038132, 34.1047211
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1561127, 25.1521835
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9057541, 29.9072723
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6200409, 44.6083145
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9692078, 35.9701233
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1122131, 36.1060791
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8284149, 39.8223419
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0169983, 45.0080032
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9906921, 48.9856262
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4192200, 48.4085770
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1497040, 39.1515503
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0811157, 38.0813446
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1833572, 23.1835632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1431

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7673368, upper bound: 13.7970794
time: 25.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7722860, upper bound: 13.7921298
time: 43.60 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 71.08 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7862399, upper bound: 13.7918228
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7771369, upper bound: 13.8009270
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7887030, upper bound: 13.7500140
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7960891, upper bound: 13.7426206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7814058, upper bound: 13.7970088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7769874, upper bound: 13.8022649
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7821263, upper bound: 13.7988682
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7792311, upper bound: 13.8017644
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7673368, upper bound: 13.7970794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 71.08
Output dim: 1, lower bound: -13.7722860, upper bound: 13.7921298

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0318832, 31.0320168
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4986191, 21.5014801
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0385132, 21.0374641
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1017151, 20.0951729
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8042679, 25.8068237
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9945831, 23.9901924
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8483505, 29.8468475
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9025917, 25.9005318
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4248962, 29.4261856
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9532089, 27.9567642
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8195038, 39.8258591
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8180237, 29.8159752
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5657501, 32.5733566
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6264572, 38.6164551
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1163483, 57.1293182
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1337433, 28.1360931
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7272415, 29.7311707
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5003815, 51.5028381
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1289673, 33.1438675
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0853271, 22.0784874
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4935722, 22.4930725
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0301895, 29.0233002
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5671844, 27.5640411
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1261139, 24.1239891
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9028625, 25.9022827
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0338898, 28.0288696
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4184036, 33.4279785
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4279022, 26.4279404
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7097244, 28.7043610
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0954285, 34.0946350
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1453934, 25.1392136
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9015732, 29.9028320
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6080322, 44.5937958
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9647369, 35.9696350
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1039963, 36.0965576
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8226852, 39.8186111
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0168152, 45.0112000
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9870300, 48.9842453
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4091339, 48.3961182
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1554108, 39.1573563
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0770721, 38.0773926
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1848602, 23.1835594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 878

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7595092, upper bound: 13.8021219
time: 32.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7768442, upper bound: 13.7847829
time: 32.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0315704, 31.0316315
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4983330, 21.5016289
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0412979, 21.0415154
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0999107, 20.0996628
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.8057861, 25.8078918
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9914894, 23.9895210
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8517990, 29.8527451
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9001465, 25.9008675
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4236908, 29.4270058
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9522858, 27.9553146
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8229904, 39.8265915
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8172760, 29.8165512
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5804367, 32.5827713
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6315842, 38.6166306
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1445160, 57.1480713
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1347351, 28.1342392
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7265244, 29.7299538
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5185699, 51.5149384
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1506577, 33.1599808
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0843277, 22.0790634
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4947815, 22.4969215
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0290756, 29.0245132
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5707397, 27.5671082
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1251221, 24.1245003
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9048538, 25.9060135
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0395584, 28.0360260
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4283981, 33.4350052
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4267044, 26.4291611
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7102890, 28.7048111
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0939789, 34.0963516
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1460953, 25.1414719
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9041977, 29.9061432
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6178894, 44.6094971
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9704285, 35.9747849
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1104813, 36.1041107
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8248596, 39.8172531
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0175171, 45.0091171
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9875641, 48.9829178
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4145660, 48.4002075
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1547699, 39.1578140
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0792694, 38.0795975
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1874199, 23.1892853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 532

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7707942, upper bound: 13.7993933
time: 27.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7768181, upper bound: 13.7933012
time: 31.20 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 60.35 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.35
Output dim: 1, lower bound: -13.7595092, upper bound: 13.8021219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.35
Output dim: 1, lower bound: -13.7768442, upper bound: 13.7847829
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.35
Output dim: 1, lower bound: -13.7707942, upper bound: 13.7993933
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.35
Output dim: 1, lower bound: -13.7768181, upper bound: 13.7933012

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0176315, 31.0199623
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4799156, 21.4857750
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0178223, 21.0202370
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0759659, 20.0755310
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7857552, 25.7913933
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9671288, 23.9673386
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8527985, 29.8510284
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8863068, 25.8869591
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4240570, 29.4275665
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9481506, 27.9518051
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8246460, 39.8283997
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8065872, 29.8022385
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5850220, 32.5888481
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6034164, 38.5968475
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1023254, 57.1121674
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1418381, 28.1478882
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7311707, 29.7348595
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.4914169, 51.4901886
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1132202, 33.1249390
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0642395, 22.0531273
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4699478, 22.4647408
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0151367, 29.0052147
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5544205, 27.5489883
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1071854, 24.1012878
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8881073, 25.8845825
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0090942, 27.9991302
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3966064, 33.4018173
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4046326, 26.4000244
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7034912, 28.6971207
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0764008, 34.0718765
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1229095, 25.1122513
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9045258, 29.9052734
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6123199, 44.5978928
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9641571, 35.9690247
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1045380, 36.0986481
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8277283, 39.8241959
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0402222, 45.0306015
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9868164, 48.9840393
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4132690, 48.4000931
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1718292, 39.1713257
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0774841, 38.0777893
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1779137, 23.1745491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1728

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1727

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7518120, upper bound: 13.8018235
time: 28.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7592111, upper bound: 13.7944214
time: 32.06 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 62.72 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 62.72
Output dim: 1, lower bound: -13.7518120, upper bound: 13.8018235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 62.72
Output dim: 1, lower bound: -13.7592111, upper bound: 13.7944214

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0283241, 31.0289192
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4811363, 21.4869041
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0090218, 21.0127411
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0635567, 20.0651474
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7726097, 25.7802734
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9481506, 23.9509201
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8608398, 29.8624039
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8659439, 25.8694458
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4172516, 29.4216728
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9458160, 27.9496574
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8143311, 39.8196487
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8144226, 29.8115463
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5874252, 32.5897064
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5759735, 38.5730896
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1150970, 57.1221924
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1427307, 28.1485901
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7155914, 29.7216759
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.4990616, 51.4960327
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.0855637, 33.0923462
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0584641, 22.0466270
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4634476, 22.4573364
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0034256, 28.9917526
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5394058, 27.5317802
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0855789, 24.0762367
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8769226, 25.8717880
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9907074, 27.9778633
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3672256, 33.3676910
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3839569, 26.3762550
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6964874, 28.6892090
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0767212, 34.0721817
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1227188, 25.1120491
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8908463, 29.8939896
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6094208, 44.5951385
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9663773, 35.9714279
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1054077, 36.0994949
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8385925, 39.8365555
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0485077, 45.0371094
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9848633, 48.9819641
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3833618, 48.3748627
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1551514, 39.1569748
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0730438, 38.0748444
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1777267, 23.1743622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 851

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7328616, upper bound: 13.8005539
time: 39.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7505407, upper bound: 13.7828657
time: 34.22 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 76.20 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 76.20
Output dim: 1, lower bound: -13.7328616, upper bound: 13.8005539
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 76.20
Output dim: 1, lower bound: -13.7505407, upper bound: 13.7828657

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 53.97 + 2141.02 = 2194.99 seconds
