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
execution time: IAR + RelationalAnalysis = 2.91 + 51.64 = 54.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -13.8153362, upper bound: 13.8153362

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 28.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 27.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 55.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 55.91
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
IS_A2, status: Status.UNKNOWN, split count: 1, time: 55.91
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.1053114, 25.9394035, -8.1601429, 25.9406853, -30.9478531, 31.0004158
1: -0.3961701, 26.8047485, -0.4283967, 26.8064613, -21.4703140, 21.5004692
2: -0.4330673, 25.8413696, -0.4901562, 25.8443508, -20.9636726, 21.0176086
3: -4.7635269, 22.5311508, -4.8289976, 22.5323067, -20.0345345, 20.0907669
4: -7.7039061, 22.5475311, -7.7519870, 22.5512390, -25.7421074, 25.7879448
5: -4.8648133, 24.8625603, -4.9317393, 24.8646660, -23.8876534, 23.9508591
6: -39.4298935, -4.1215267, -39.4362717, -4.1110477, -29.8414001, 29.8343430
7: -9.2394352, 23.4325981, -9.2776461, 23.4345970, -25.8622589, 25.8984337
8: -13.9319925, 20.0172119, -13.9645338, 20.0224743, -29.3862762, 29.4145164
9: -8.6302691, 22.5547733, -8.6414557, 22.5598526, -27.9392014, 27.9328537
10: -29.0440102, 17.9230118, -29.0493851, 17.9623833, -39.8217545, 39.7855148
11: -26.2461777, 6.8615055, -26.2510529, 6.9229407, -29.7956085, 29.7354889
12: -46.0995865, -8.2859230, -46.1024170, -8.2027493, -32.5226593, 32.4463730
13: -32.5963821, 13.2570305, -32.6318245, 13.2681484, -38.5924911, 38.6144485
14: -59.5300026, -2.1005116, -59.5478745, -2.0125923, -57.0651703, 57.0015869
15: -14.2677231, 18.8320999, -14.2989674, 18.8350201, -28.1052475, 28.1297874
16: -15.6943283, 22.3006401, -15.7091770, 22.3166389, -29.7350998, 29.7304153
17: -59.2464561, -6.9026995, -59.2555122, -6.8429365, -51.4951630, 51.4439850
18: -22.0605659, 16.3363724, -22.0664177, 16.3843670, -33.1547546, 33.1322479
19: -22.1717949, 6.0634527, -22.1804962, 6.1132956, -22.0794601, 22.0375519
20: -27.8478165, 0.6519361, -27.8550110, 0.7038112, -22.4850807, 22.4405746
21: -26.3245754, 7.4243827, -26.3342762, 7.4917402, -29.0028152, 28.9435349
22: -29.4903278, 5.3207140, -29.4978848, 5.3472228, -27.5783310, 27.5575256
23: -17.8390274, 12.3138142, -17.8462925, 12.3613625, -24.1172867, 24.0773659
24: -16.5010185, 13.6555300, -16.5057316, 13.6891432, -25.9054718, 25.8787689
25: -23.8391113, 8.9234428, -23.8433819, 8.9601040, -28.0529175, 28.0221901
26: -39.3425560, 4.0884829, -39.3480301, 4.1845145, -33.3980560, 33.3072052
27: -19.5118561, 14.8983860, -19.5232658, 14.9358616, -34.4477158, 34.4216537
28: -21.3914585, 11.4891376, -21.4019260, 11.5451231, -26.4068298, 26.3621407
29: -24.4092560, 6.4216204, -24.4175663, 6.4498529, -28.6766129, 28.6503677
30: -30.0218391, 5.8784432, -30.0269966, 5.9355941, -34.0640411, 34.0105972
31: -23.3667107, 7.6948080, -23.3764572, 7.7376213, -25.1471329, 25.1139603
32: -37.0023041, -2.5707016, -37.0080566, -2.5549550, -29.8859787, 29.8735275
33: -54.4506073, 0.1120796, -54.4960022, 0.1238394, -44.5733643, 44.6083298
34: -49.1674805, -6.9501758, -49.1768761, -6.9447927, -35.9538727, 35.9476471
35: -40.3191757, 3.9486217, -40.3333893, 3.9505339, -36.1216583, 36.1293411
36: -45.6260338, 1.3915272, -45.6307716, 1.4000950, -39.8349304, 39.8241882
37: -60.7357368, -9.8311396, -60.7401352, -9.8062906, -45.0176849, 45.0039368
38: -53.6797256, 4.1373672, -53.6847649, 4.1476822, -48.9745331, 48.9777603
39: -61.9510574, -4.5284710, -61.9689026, -4.5146942, -48.4161835, 48.4196243
40: -50.1949959, -9.1540918, -50.2014656, -9.1490717, -39.1394501, 39.1390305
41: -32.0110168, 7.3371124, -32.0196075, 7.3495636, -38.0528412, 38.0482101
42: -30.1158543, -0.2735119, -30.1209259, -0.2600203, -23.1749802, 23.1644402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=192, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 39.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
time: 29.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.2225857, 26.0007324, -8.1999702, 25.9415169, -31.0511856, 31.1061249
1: -0.4706707, 26.8796654, -0.4518881, 26.8075409, -21.5377197, 21.5999336
2: -0.5409107, 25.9500332, -0.5318675, 25.8464317, -21.0602036, 21.1677704
3: -4.8847761, 22.6493950, -4.8773403, 22.5328770, -20.1152916, 20.2119446
4: -7.8004246, 22.6440086, -7.7878141, 22.5537434, -25.8270416, 25.9212189
5: -4.9906955, 24.9724998, -4.9808216, 24.8660297, -23.9851227, 24.1000443
6: -39.4593430, -4.1008406, -39.4407654, -4.1106730, -29.8805237, 29.8595200
7: -9.3306875, 23.4828110, -9.3063374, 23.4357948, -25.9456482, 25.9738464
8: -13.9971371, 20.1027679, -13.9877262, 20.0260620, -29.4527321, 29.5256081
9: -8.6547222, 22.5765114, -8.6477833, 22.5626259, -28.0108109, 27.9424286
10: -29.1166382, 18.0099068, -29.0530090, 17.9873886, -39.9423828, 39.8769150
11: -26.4747162, 6.9742451, -26.2542610, 6.9700027, -30.0714264, 29.8163681
12: -46.2203293, -8.1291113, -46.1043282, -8.1417542, -32.6998596, 32.5731888
13: -32.6291656, 13.3474455, -32.6387405, 13.2755785, -38.6346588, 38.7077484
14: -59.7450218, -1.9441128, -59.5604477, -1.9475460, -57.3345795, 57.1192017
15: -14.3006802, 18.8801003, -14.2986851, 18.8367119, -28.1561050, 28.1692047
16: -15.7794685, 22.3137779, -15.7196503, 22.3100796, -29.8241272, 29.7535706
17: -59.4829178, -6.7918291, -59.2615700, -6.7979345, -51.7860947, 51.5258484
18: -22.2366257, 16.4182167, -22.0698090, 16.4188385, -33.3074951, 33.1787415
19: -22.3293304, 6.1516886, -22.1865082, 6.1502213, -22.2774467, 22.1075783
20: -27.9729633, 0.7454853, -27.8600998, 0.7420688, -22.6591873, 22.5148659
21: -26.5225601, 7.5442829, -26.3409805, 7.5414948, -29.2551804, 29.0448761
22: -29.5889015, 5.3724508, -29.5031433, 5.3652062, -27.7016373, 27.6120567
23: -17.9811668, 12.4026117, -17.8514481, 12.3963432, -24.2972870, 24.1484222
24: -16.5836639, 13.7159328, -16.5088863, 13.7139845, -26.0061493, 25.9238739
25: -23.9264679, 8.9900522, -23.8462563, 8.9863186, -28.1635208, 28.0729980
26: -39.5498924, 4.2547207, -39.3517570, 4.2555757, -33.6876984, 33.4377060
27: -19.6157379, 14.9654865, -19.5311165, 14.9650373, -34.5807762, 34.4966049
28: -21.5402641, 11.5897226, -21.4094238, 11.5863380, -26.5985336, 26.4444351
29: -24.5351238, 6.4744215, -24.4233112, 6.4695997, -28.8441010, 28.7070389
30: -30.1812115, 5.9900088, -30.0305920, 5.9775443, -34.2645569, 34.1046371
31: -23.5170383, 7.7707691, -23.3834667, 7.7692113, -25.3330841, 25.1711197
32: -37.0256805, -2.5361004, -37.0118484, -2.5479898, -29.9222260, 29.9120560
33: -54.5336189, 0.2526608, -54.5291367, 0.1323576, -44.6374664, 44.7973633
34: -49.1928406, -6.9099126, -49.1808090, -6.9408689, -36.0160828, 35.9788513
35: -40.3351974, 3.9856234, -40.3351784, 3.9518919, -36.1530151, 36.1884155
36: -45.6520195, 1.4236488, -45.6337471, 1.4051332, -39.8964844, 39.8615036
37: -60.7890625, -9.7930803, -60.7428246, -9.7976303, -45.0556335, 45.0708771
38: -53.7174644, 4.1758327, -53.6881104, 4.1508818, -48.9934387, 49.0715942
39: -61.9881821, -4.4242554, -61.9814949, -4.5046940, -48.4578247, 48.5443268
40: -50.2338257, -9.1107578, -50.2055855, -9.1457481, -39.1679382, 39.2080383
41: -32.0445251, 7.3659415, -32.0253868, 7.3525004, -38.0909119, 38.0826874
42: -30.1373596, -0.2405729, -30.1246033, -0.2570233, -23.2112656, 23.1982498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=192, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 26.14 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8106329, upper bound: 13.8106329
time: 28.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 56.83 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 56.83
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 56.83
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 56.83
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 56.83
Output dim: 1, lower bound: -13.8106329, upper bound: 13.8106329

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.1031246, 25.9393272, -8.1538296, 25.9404640, -30.9454269, 30.9909172
1: -0.3946671, 26.8046989, -0.4241385, 26.8063202, -21.4686966, 21.4808578
2: -0.4316597, 25.8413467, -0.4861965, 25.8441963, -20.9621582, 20.9908371
3: -4.7624640, 22.5309944, -4.8259254, 22.5318222, -20.0329704, 20.0708961
4: -7.7021379, 22.5474644, -7.7469716, 22.5509911, -25.7400665, 25.7576981
5: -4.8636351, 24.8624687, -4.9284043, 24.8644409, -23.8861694, 23.9278679
6: -39.4297295, -4.1231813, -39.4358826, -4.1159010, -29.8310013, 29.8321991
7: -9.2381201, 23.4325657, -9.2739239, 23.4344559, -25.8606796, 25.8720703
8: -13.9298420, 20.0170708, -13.9583826, 20.0220680, -29.3837280, 29.3725815
9: -8.6290226, 22.5546799, -8.6377888, 22.5594940, -27.9375305, 27.9125633
10: -29.0422440, 17.9227791, -29.0445366, 17.9617424, -39.8194275, 39.7436218
11: -26.2458115, 6.8605309, -26.2499619, 6.9201794, -29.7863617, 29.7333870
12: -46.0993614, -8.2869892, -46.1018066, -8.2058773, -32.5275650, 32.4422302
13: -32.5958405, 13.2559443, -32.6301575, 13.2648478, -38.6421814, 38.6029053
14: -59.5281906, -2.1007271, -59.5426712, -2.0131893, -57.0655975, 56.9952087
15: -14.2660236, 18.8319893, -14.2942295, 18.8344955, -28.1031647, 28.1202850
16: -15.6928043, 22.3005886, -15.7050362, 22.3165131, -29.7336502, 29.6803513
17: -59.2450867, -6.9029226, -59.2517853, -6.8436184, -51.5098495, 51.4367676
18: -22.0601788, 16.3356819, -22.0651588, 16.3823891, -33.1493149, 33.1452408
19: -22.1715336, 6.0624228, -22.1797371, 6.1102829, -22.0677948, 22.0359001
20: -27.8476620, 0.6506319, -27.8545685, 0.7001448, -22.4652214, 22.4388695
21: -26.3241844, 7.4229569, -26.3332520, 7.4876270, -28.9848328, 28.9410095
22: -29.4901123, 5.3195958, -29.4972019, 5.3440924, -27.5450058, 27.5556946
23: -17.8387909, 12.3129549, -17.8457108, 12.3589296, -24.1010742, 24.0759315
24: -16.5008850, 13.6546211, -16.5052605, 13.6865931, -25.8932571, 25.8773727
25: -23.8390121, 8.9222145, -23.8430138, 8.9565563, -28.0151062, 28.0202789
26: -39.3423004, 4.0873904, -39.3474350, 4.1812782, -33.3850784, 33.3055420
27: -19.5115547, 14.8973637, -19.5223141, 14.9328537, -34.4444084, 34.4196777
28: -21.3913097, 11.4878607, -21.4014435, 11.5415163, -26.3920441, 26.3603783
29: -24.4089870, 6.4209175, -24.4167404, 6.4477606, -28.6503067, 28.6488609
30: -30.0216484, 5.8770871, -30.0264645, 5.9316902, -34.0511932, 34.0087280
31: -23.3664856, 7.6935940, -23.3757343, 7.7341824, -25.1375275, 25.1121407
32: -37.0021515, -2.5718336, -37.0076065, -2.5582490, -29.8696060, 29.8718796
33: -54.4503899, 0.1100492, -54.4953461, 0.1178942, -44.5129242, 44.6057205
34: -49.1672974, -6.9517198, -49.1762390, -6.9492931, -35.9330139, 35.9456711
35: -40.3189354, 3.9467392, -40.3327408, 3.9451265, -36.0768280, 36.1270676
36: -45.6258926, 1.3896751, -45.6302948, 1.3946581, -39.7813873, 39.8218765
37: -60.7354736, -9.8323593, -60.7394257, -9.8097630, -44.9721375, 45.0018005
38: -53.6794739, 4.1349602, -53.6840591, 4.1409044, -48.9017639, 48.9747314
39: -61.9508438, -4.5304031, -61.9682312, -4.5200291, -48.3579102, 48.4171371
40: -50.1947327, -9.1550474, -50.2006531, -9.1518269, -39.1352539, 39.1465454
41: -32.0108452, 7.3360004, -32.0190392, 7.3463626, -38.0482941, 38.0465317
42: -30.1157227, -0.2742271, -30.1205063, -0.2620125, -23.1679459, 23.1617737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7731175, upper bound: 13.7900057
time: 30.71 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7731175, upper bound: 13.8093527
time: 37.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.2204018, 26.0006714, -8.1936712, 25.9413071, -31.0487747, 31.0966187
1: -0.4692087, 26.8796215, -0.4476295, 26.8074036, -21.5360947, 21.5802765
2: -0.5395069, 25.9499969, -0.5279360, 25.8462944, -21.0586891, 21.1409454
3: -4.8836961, 22.6492233, -4.8742690, 22.5324402, -20.1136971, 20.1920738
4: -7.7986979, 22.6439552, -7.7828155, 22.5535011, -25.8250427, 25.8910141
5: -4.9895344, 24.9723854, -4.9774647, 24.8657799, -23.9836426, 24.0770645
6: -39.4592361, -4.1025162, -39.4403419, -4.1154957, -29.8701401, 29.8573990
7: -9.3293762, 23.4827518, -9.3025608, 23.4356270, -25.9440804, 25.9474907
8: -13.9950037, 20.1026039, -13.9815521, 20.0256538, -29.4502182, 29.4837189
9: -8.6534672, 22.5763931, -8.6441336, 22.5622673, -28.0091553, 27.9221649
10: -29.1148872, 18.0097027, -29.0482254, 17.9867439, -39.9399872, 39.8349991
11: -26.4743176, 6.9732966, -26.2532463, 6.9672198, -30.0622025, 29.8142853
12: -46.2200623, -8.1301889, -46.1037369, -8.1449223, -32.7047729, 32.5690193
13: -32.6285286, 13.3463106, -32.6370316, 13.2723141, -38.6844025, 38.6962280
14: -59.7432060, -1.9443398, -59.5551529, -1.9481401, -57.3351288, 57.1128845
15: -14.2990379, 18.8799133, -14.2939682, 18.8361893, -28.1540222, 28.1597023
16: -15.7779274, 22.3137093, -15.7155380, 22.3099613, -29.8226624, 29.7035027
17: -59.4814720, -6.7920685, -59.2579346, -6.7985458, -51.8007278, 51.5186157
18: -22.2362270, 16.4175835, -22.0685616, 16.4168606, -33.3020248, 33.1917419
19: -22.3290615, 6.1506248, -22.1857491, 6.1471882, -22.2657700, 22.1059189
20: -27.9727974, 0.7442088, -27.8596573, 0.7383728, -22.6393127, 22.5131607
21: -26.5221291, 7.5427923, -26.3399162, 7.5373578, -29.2372131, 29.0423737
22: -29.5886860, 5.3713474, -29.5024643, 5.3620758, -27.6683044, 27.6102448
23: -17.9809399, 12.4017944, -17.8508778, 12.3938875, -24.2811203, 24.1470261
24: -16.5835419, 13.7150364, -16.5084190, 13.7114267, -25.9939423, 25.9225159
25: -23.9263535, 8.9887905, -23.8459282, 8.9827557, -28.1257172, 28.0710335
26: -39.5497208, 4.2536159, -39.3511543, 4.2523694, -33.6747284, 33.4360657
27: -19.6154156, 14.9644165, -19.5301895, 14.9620295, -34.5774460, 34.4946060
28: -21.5400772, 11.5884724, -21.4089088, 11.5826864, -26.5837479, 26.4426765
29: -24.5348167, 6.4736786, -24.4224625, 6.4675074, -28.8177719, 28.7055321
30: -30.1809998, 5.9886627, -30.0300655, 5.9736280, -34.2517166, 34.1027832
31: -23.5167732, 7.7695808, -23.3827076, 7.7657714, -25.3235016, 25.1693077
32: -37.0254898, -2.5373282, -37.0114365, -2.5513258, -29.9058228, 29.9104004
33: -54.5334091, 0.2505827, -54.5284653, 0.1263876, -44.5770721, 44.7947617
34: -49.1926384, -6.9114695, -49.1802025, -6.9453502, -35.9952621, 35.9768448
35: -40.3349915, 3.9837103, -40.3345490, 3.9465179, -36.1081848, 36.1860886
36: -45.6518135, 1.4217672, -45.6333313, 1.3997145, -39.8428726, 39.8591690
37: -60.7887497, -9.7942238, -60.7420731, -9.8011312, -45.0100403, 45.0687714
38: -53.7172661, 4.1734552, -53.6874084, 4.1439743, -48.9206543, 49.0686264
39: -61.9879646, -4.4261847, -61.9808655, -4.5100679, -48.3995819, 48.5418777
40: -50.2335663, -9.1117287, -50.2047310, -9.1484833, -39.1637421, 39.2155228
41: -32.0443459, 7.3648281, -32.0247726, 7.3493023, -38.0863342, 38.0810394
42: -30.1372337, -0.2412095, -30.1242027, -0.2590361, -23.2042618, 23.1955986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8093528, upper bound: 13.7900057
time: 46.32 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8093528, upper bound: 13.8093527
time: 35.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 84.17 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 84.17
Output dim: 1, lower bound: -13.7731175, upper bound: 13.7900057
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 84.17
Output dim: 1, lower bound: -13.7731175, upper bound: 13.8093527
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 84.17
Output dim: 1, lower bound: -13.8093528, upper bound: 13.7900057
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 84.17
Output dim: 1, lower bound: -13.8093528, upper bound: 13.8093527

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.1006012, 25.9392300, -8.1485796, 25.9402542, -30.9430389, 30.9630356
1: -0.3925395, 26.8046074, -0.4194040, 26.8061199, -21.4668045, 21.4694710
2: -0.4290676, 25.8412533, -0.4804058, 25.8439903, -20.9595261, 20.9555740
3: -4.7595701, 22.5308056, -4.8195305, 22.5314178, -20.0307617, 19.9963913
4: -7.6996794, 22.5472527, -7.7414904, 22.5505753, -25.7368584, 25.7088394
5: -4.8605037, 24.8621941, -4.9213052, 24.8638153, -23.8834724, 23.8570023
6: -39.4294205, -4.1256027, -39.4352837, -4.1212816, -29.8212051, 29.8293304
7: -9.2341928, 23.4324646, -9.2659178, 23.4342098, -25.8616447, 25.8649521
8: -13.9274912, 20.0167961, -13.9531889, 20.0215034, -29.3811531, 29.3598747
9: -8.6285162, 22.5528145, -8.6367235, 22.5553226, -27.9290848, 27.9252739
10: -29.0420055, 17.9198341, -29.0439568, 17.9552536, -39.7990494, 39.7401123
11: -26.2454987, 6.8572273, -26.2493992, 6.9127417, -29.6739998, 29.7298737
12: -46.0991669, -8.2921486, -46.1014137, -8.2172737, -32.4569244, 32.4369049
13: -32.5925369, 13.2551498, -32.6227722, 13.2632456, -38.6267471, 38.6251907
14: -59.5272217, -2.1048222, -59.5405807, -2.0223255, -56.9770050, 56.9890594
15: -14.2587051, 18.8316841, -14.2776947, 18.8339386, -28.0936661, 28.0754395
16: -15.6919003, 22.2968826, -15.7030563, 22.3081017, -29.7127838, 29.6743622
17: -59.2446442, -6.9043760, -59.2508736, -6.8468390, -51.4678650, 51.4336395
18: -22.0597286, 16.3348885, -22.0642700, 16.3806248, -33.1290359, 33.1366882
19: -22.1710968, 6.0609121, -22.1787548, 6.1069555, -22.0168648, 22.0334091
20: -27.8472862, 0.6486268, -27.8536282, 0.6956558, -22.3897324, 22.4365807
21: -26.3237152, 7.4205537, -26.3321934, 7.4822946, -28.9044647, 28.9385986
22: -29.4898186, 5.3162570, -29.4965477, 5.3365202, -27.5338211, 27.5729866
23: -17.8383789, 12.3114986, -17.8448315, 12.3555775, -24.0557327, 24.0709534
24: -16.5004978, 13.6524534, -16.5044899, 13.6819878, -25.9073563, 25.8688507
25: -23.8387299, 8.9208679, -23.8424606, 8.9535465, -27.9874344, 28.0176010
26: -39.3419685, 4.0849166, -39.3467255, 4.1758480, -33.2922516, 33.3023224
27: -19.5109024, 14.8941574, -19.5209389, 14.9270115, -34.4379120, 34.4150963
28: -21.3907413, 11.4862375, -21.4001923, 11.5378771, -26.3422318, 26.3574791
29: -24.4086685, 6.4179001, -24.4161453, 6.4419155, -28.6423798, 28.6618080
30: -30.0213757, 5.8742971, -30.0258827, 5.9262161, -34.0112457, 34.0048218
31: -23.3659649, 7.6924925, -23.3746586, 7.7316823, -25.0995636, 25.1080551
32: -37.0019684, -2.5733576, -37.0071716, -2.5616016, -29.8576202, 29.8703690
33: -54.4483185, 0.1095753, -54.4908257, 0.1169777, -44.5106354, 44.5396500
34: -49.1653633, -6.9520426, -49.1720352, -6.9499035, -35.9281921, 35.9219666
35: -40.3161926, 3.9466562, -40.3266792, 3.9449453, -36.0719376, 36.1046295
36: -45.6256561, 1.3873901, -45.6297531, 1.3896542, -39.7776337, 39.8283920
37: -60.7349968, -9.8350468, -60.7383652, -9.8158979, -44.9590607, 44.9954224
38: -53.6791267, 4.1324453, -53.6832848, 4.1351910, -48.9019470, 48.9650650
39: -61.9496384, -4.5310564, -61.9655838, -4.5214310, -48.3522491, 48.4103088
40: -50.1941605, -9.1553068, -50.1994858, -9.1523895, -39.1425323, 39.1385498
41: -32.0104599, 7.3343706, -32.0181808, 7.3426809, -38.0448151, 38.0424805
42: -30.1155243, -0.2778702, -30.1200638, -0.2701025, -23.1496506, 23.1560059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7507197, upper bound: 13.8080303
time: 38.45 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
time: 28.78 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.1613102, 25.9996204, -8.0673532, 25.9211006, -30.9673996, 30.9688759
1: -0.4360771, 26.8784046, -0.3748679, 26.7821388, -21.4735565, 21.5019302
2: -0.4779158, 25.9479847, -0.4014416, 25.8055916, -20.9562454, 21.0152397
3: -4.8146739, 22.6469975, -4.7336082, 22.4814816, -19.9871826, 20.0442123
4: -7.7423468, 22.6408138, -7.6650305, 22.5103722, -25.7279243, 25.7719879
5: -4.9135532, 24.9683437, -4.8219004, 24.8147964, -23.8539772, 23.9152031
6: -39.4525146, -4.1135044, -39.4198723, -4.1436520, -29.8324966, 29.8215561
7: -9.2805986, 23.4802322, -9.1938639, 23.4197540, -25.8613510, 25.8322296
8: -13.9470320, 20.0983086, -13.8811712, 19.9807167, -29.3554459, 29.3786049
9: -8.6437788, 22.5533600, -8.6159077, 22.5105476, -27.9451065, 27.8488426
10: -29.1086178, 17.9534416, -28.9977169, 17.8661499, -39.8146667, 39.7309113
11: -26.4703693, 6.9241443, -26.1727715, 6.8686967, -29.9520569, 29.6757736
12: -46.2168236, -8.2498379, -46.0279121, -8.3890667, -32.4598770, 32.3750839
13: -32.6162949, 13.3302402, -32.6104431, 13.2301970, -38.6298523, 38.6353989
14: -59.7233238, -2.0445852, -59.4389877, -2.1513157, -57.1119690, 56.8972626
15: -14.2692375, 18.8772659, -14.2261181, 18.8133411, -28.0973587, 28.0855446
16: -15.7629757, 22.3007870, -15.6658859, 22.2797852, -29.7763901, 29.6383591
17: -59.4737854, -6.8478994, -59.1668167, -6.9147968, -51.6792221, 51.3742218
18: -22.2305183, 16.3827438, -22.0017490, 16.3492966, -33.2384186, 33.1341095
19: -22.3193398, 6.1148124, -22.1232758, 6.0750780, -22.1828194, 22.0017052
20: -27.9648495, 0.6952138, -27.8026123, 0.6386952, -22.5216904, 22.3951454
21: -26.5130157, 7.4852643, -26.2611847, 7.4208770, -29.1030273, 28.8944855
22: -29.5816669, 5.3503695, -29.4763603, 5.3134785, -27.6052170, 27.5175705
23: -17.9727478, 12.3675795, -17.7941475, 12.3232727, -24.2040253, 24.0682945
24: -16.5768566, 13.7043133, -16.4888973, 13.6883259, -25.9580078, 25.8949738
25: -23.9219418, 8.9634304, -23.8174095, 8.9295244, -28.0725250, 28.0253601
26: -39.5442619, 4.1559234, -39.2637711, 4.0585518, -33.4721680, 33.2511139
27: -19.6019211, 14.9533043, -19.4942932, 14.9392872, -34.5412064, 34.4475975
28: -21.5285778, 11.5498362, -21.3511543, 11.5037613, -26.4939270, 26.3494949
29: -24.5275555, 6.4504547, -24.3823318, 6.4165630, -28.7535477, 28.6110916
30: -30.1771355, 5.9555426, -29.9887447, 5.9022317, -34.1766968, 34.0340424
31: -23.5049286, 7.7433729, -23.3207150, 7.7122355, -25.2640991, 25.0921364
32: -37.0205841, -2.5691929, -36.9887543, -2.6195245, -29.8331757, 29.8504333
33: -54.4889679, 0.2386131, -54.4364777, 0.0594616, -44.4674683, 44.6917648
34: -49.1727715, -6.9175663, -49.1357460, -6.9824791, -35.9476013, 35.9244385
35: -40.3183975, 3.9818058, -40.2985992, 3.9270496, -36.0660095, 36.1425629
36: -45.6467667, 1.4067230, -45.6175423, 1.3600559, -39.7882538, 39.8049316
37: -60.7814636, -9.8208694, -60.7062836, -9.8591747, -44.9303894, 45.0116348
38: -53.7104530, 4.1538601, -53.6565247, 4.0946846, -48.8531952, 49.0288086
39: -61.9782639, -4.4414549, -61.9568558, -4.5583391, -48.3312073, 48.4938202
40: -50.2231331, -9.1172638, -50.1708870, -9.1700535, -39.1131744, 39.1781845
41: -32.0348549, 7.3537726, -31.9970646, 7.3228974, -38.0474243, 38.0397568
42: -30.1319046, -0.2586703, -30.1087418, -0.3004289, -23.1549759, 23.1539154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
time: 36.83 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085879, upper bound: 13.7892414
time: 31.60 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.2180004, 26.0005608, -8.1886101, 25.9411011, -31.0464973, 31.0689659
1: -0.4671631, 26.8795319, -0.4430695, 26.8071995, -21.5342674, 21.5691032
2: -0.5370388, 25.9498940, -0.5223694, 25.8460922, -21.0562325, 21.1059647
3: -4.8809433, 22.6490250, -4.8680539, 22.5320358, -20.1115799, 20.1177750
4: -7.7963533, 22.6437569, -7.7775979, 22.5531235, -25.8219147, 25.8424149
5: -4.9865360, 24.9720955, -4.9706230, 24.8651924, -23.9810944, 24.0064240
6: -39.4589462, -4.1048565, -39.4398079, -4.1208420, -29.8599319, 29.8546677
7: -9.3259945, 23.4826870, -9.2948551, 23.4354134, -25.9451065, 25.9406662
8: -13.9928274, 20.1023712, -13.9766150, 20.0250893, -29.4477806, 29.4712524
9: -8.6529541, 22.5746536, -8.6430988, 22.5583096, -28.0008621, 27.9349823
10: -29.1146469, 18.0069084, -29.0476837, 17.9805546, -39.9197388, 39.8316422
11: -26.4740715, 6.9700141, -26.2526264, 6.9598494, -29.9497604, 29.8108864
12: -46.2199135, -8.1349640, -46.1033249, -8.1556520, -32.6346970, 32.5640488
13: -32.6255722, 13.3456945, -32.6296654, 13.2709599, -38.6692810, 38.7108154
14: -59.7422791, -1.9481697, -59.5531960, -1.9568424, -57.2433014, 57.1068878
15: -14.2917881, 18.8797131, -14.2774410, 18.8356552, -28.1447144, 28.1087799
16: -15.7770252, 22.3100796, -15.7136421, 22.3015556, -29.8013916, 29.6977310
17: -59.4811172, -6.7934465, -59.2569695, -6.8016424, -51.7592010, 51.5156097
18: -22.2358303, 16.4167576, -22.0676880, 16.4150848, -33.2676544, 33.1839981
19: -22.3286629, 6.1491394, -22.1847935, 6.1438890, -22.2149544, 22.1034050
20: -27.9724216, 0.7422757, -27.8587894, 0.7340469, -22.5621338, 22.5110054
21: -26.5217285, 7.5404735, -26.3389130, 7.5321651, -29.1570358, 29.0400696
22: -29.5884247, 5.3684492, -29.5018768, 5.3547025, -27.6572723, 27.6275330
23: -17.9805450, 12.4003506, -17.8500175, 12.3906603, -24.2356415, 24.1421394
24: -16.5832176, 13.7129126, -16.5076790, 13.7068300, -25.9944916, 25.9149246
25: -23.9260826, 8.9874830, -23.8453941, 8.9798174, -28.0977402, 28.0685501
26: -39.5494385, 4.2513008, -39.3505135, 4.2471824, -33.5785522, 33.4329834
27: -19.6148720, 14.9620571, -19.5289803, 14.9562225, -34.5710945, 34.4910355
28: -21.5395432, 11.5868444, -21.4077721, 11.5791550, -26.5340195, 26.4397659
29: -24.5345421, 6.4711876, -24.4219322, 6.4618320, -28.8100662, 28.7185783
30: -30.1807404, 5.9858651, -30.0294933, 5.9681993, -34.2117996, 34.0988617
31: -23.5163136, 7.7684875, -23.3817139, 7.7632914, -25.2854156, 25.1653175
32: -37.0253372, -2.5386705, -37.0110397, -2.5543914, -29.8939972, 29.9090881
33: -54.5314369, 0.2501583, -54.5240517, 0.1254768, -44.5748901, 44.7268753
34: -49.1911163, -6.9117050, -49.1765862, -6.9459743, -35.9900208, 35.9510727
35: -40.3323059, 3.9836760, -40.3284683, 3.9463120, -36.1024017, 36.1514969
36: -45.6516113, 1.4196358, -45.6329002, 1.3948050, -39.8396072, 39.8655472
37: -60.7882919, -9.7969208, -60.7410431, -9.8071938, -44.9959564, 45.0622177
38: -53.7168655, 4.1709681, -53.6866722, 4.1385479, -48.9174500, 49.0587463
39: -61.9868393, -4.4267902, -61.9782677, -4.5114088, -48.3938599, 48.5333557
40: -50.2330246, -9.1119633, -50.2036896, -9.1490335, -39.1709290, 39.2076950
41: -32.0439758, 7.3632956, -32.0239792, 7.3456335, -38.0828094, 38.0771103
42: -30.1370678, -0.2447901, -30.1238060, -0.2670503, -23.1822357, 23.1901741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=191, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
time: 306.91 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880
time: 35.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 344.23 seconds
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.7507197, upper bound: 13.8080303
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.8085879, upper bound: 13.7892414
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 344.23
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.9673195, 25.9155273, -8.0824080, 25.9382210, -30.8061142, 30.8721733
1: -0.3043566, 26.7854309, -0.3756189, 26.8052673, -21.3768845, 21.4055481
2: -0.3568344, 25.8196392, -0.4438925, 25.8428478, -20.8863831, 20.8977737
3: -4.7013960, 22.5162830, -4.7897363, 22.5291767, -19.9678040, 19.9485931
4: -7.5994368, 22.5318623, -7.6916823, 22.5490570, -25.6350670, 25.6412048
5: -4.7878504, 24.8391476, -4.8842006, 24.8610096, -23.8066216, 23.7948685
6: -39.4159584, -4.2009268, -39.4309692, -4.1581602, -29.7575760, 29.7461243
7: -9.1280069, 23.4016609, -9.2121096, 23.4324036, -25.7525101, 25.7787704
8: -13.8320312, 19.9936638, -13.9044847, 20.0183620, -29.2806168, 29.2869453
9: -8.5301762, 22.5213795, -8.5886974, 22.5537930, -27.8292236, 27.8447380
10: -28.9560356, 17.8865604, -29.0005264, 17.9501419, -39.7084656, 39.6651459
11: -26.2028294, 6.8365464, -26.2284737, 6.9032445, -29.6169434, 29.6865692
12: -46.0795021, -8.3798094, -46.0975990, -8.2602730, -32.3955994, 32.3332596
13: -32.5376472, 13.2311745, -32.5955811, 13.2546787, -38.5561523, 38.5557327
14: -59.4348259, -2.1357746, -59.4949799, -2.0263233, -56.8711090, 56.8970795
15: -14.2148685, 18.8119202, -14.2580156, 18.8232040, -28.0367050, 28.0304947
16: -15.5884209, 22.2557850, -15.6528130, 22.3067780, -29.6047287, 29.5813065
17: -59.1734695, -6.9299726, -59.2156143, -6.8562164, -51.3764954, 51.3442383
18: -22.0285683, 16.2958183, -22.0551109, 16.3613815, -33.0769730, 33.0853577
19: -22.1480064, 6.0240865, -22.1720963, 6.0881348, -21.9731255, 21.9880829
20: -27.8232079, 0.6009326, -27.8497829, 0.6718826, -22.3415070, 22.3846779
21: -26.2971230, 7.3847218, -26.3228874, 7.4642754, -28.8573914, 28.8905487
22: -29.4497032, 5.2572918, -29.4890862, 5.3062444, -27.4624939, 27.5045357
23: -17.8172855, 12.2679634, -17.8400097, 12.3343248, -24.0114212, 24.0204239
24: -16.4792843, 13.6163034, -16.4996719, 13.6637840, -25.8681717, 25.8276367
25: -23.8114815, 8.8647594, -23.8387413, 8.9248886, -27.9307175, 27.9566422
26: -39.2990646, 4.0041742, -39.3410110, 4.1352358, -33.2084656, 33.2163239
27: -19.4796143, 14.8518629, -19.5101662, 14.9060698, -34.3856850, 34.3620300
28: -21.3637009, 11.4280720, -21.3955116, 11.5081310, -26.2824936, 26.2913208
29: -24.3754959, 6.3718615, -24.4066963, 6.4183311, -28.5843735, 28.6047440
30: -30.0061779, 5.8505611, -30.0196209, 5.9154706, -33.9772644, 33.9683533
31: -23.3396854, 7.6462131, -23.3672028, 7.7084990, -25.0471649, 25.0519104
32: -36.9858894, -2.6294436, -37.0024605, -2.5892844, -29.8054962, 29.8057327
33: -54.4060898, 0.0302544, -54.4844818, 0.0773907, -44.4250183, 44.4519653
34: -49.1200256, -7.0472345, -49.1683884, -6.9970183, -35.8350983, 35.8225327
35: -40.2694855, 3.8575449, -40.3218994, 3.9001780, -35.9816589, 36.0111923
36: -45.5724716, 1.2849274, -45.6246262, 1.3383875, -39.6724625, 39.7168884
37: -60.6885376, -9.9097652, -60.7317581, -9.8539848, -44.8723145, 44.9114990
38: -53.6074333, 3.9949150, -53.6762848, 4.0675039, -48.7597504, 48.8135529
39: -61.9055710, -4.5950289, -61.9544373, -4.5531063, -48.2753143, 48.3327026
40: -50.1692619, -9.1927395, -50.1903152, -9.1703835, -39.0949707, 39.0883026
41: -31.9803677, 7.2670970, -32.0130157, 7.3093300, -37.9775848, 37.9686127
42: -30.0978451, -0.3415022, -30.1172676, -0.3017597, -23.1041794, 23.0867500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7325000, upper bound: 13.8062907
time: 28.46 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7489824, upper bound: 13.8062907
time: 32.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.0936031, 25.9388885, -8.1455040, 25.9401016, -30.9332352, 30.9594574
1: -0.3881235, 26.8043804, -0.4174738, 26.8060112, -21.4482918, 21.4672508
2: -0.4252138, 25.8410244, -0.4787006, 25.8439007, -20.9386101, 20.9536591
3: -4.7561426, 22.5304832, -4.8178649, 22.5312710, -20.0106659, 19.9947128
4: -7.6943121, 22.5468369, -7.7391124, 22.5503845, -25.7110443, 25.7059097
5: -4.8567915, 24.8616581, -4.9197044, 24.8635712, -23.8598862, 23.8547745
6: -39.4288597, -4.1300192, -39.4350548, -4.1232576, -29.8243103, 29.8226547
7: -9.2289658, 23.4319553, -9.2636232, 23.4339752, -25.8252258, 25.8621330
8: -13.9226456, 20.0162506, -13.9510927, 20.0212440, -29.3569336, 29.3570900
9: -8.6237888, 22.5525227, -8.6346874, 22.5552044, -27.9040909, 27.9228134
10: -29.0376587, 17.9188995, -29.0421047, 17.9548340, -39.7696762, 39.7372437
11: -26.2418900, 6.8557291, -26.2478027, 6.9120679, -29.6738701, 29.7255058
12: -46.0984344, -8.2966785, -46.1010933, -8.2191935, -32.4514313, 32.4278526
13: -32.5891190, 13.2539892, -32.6211891, 13.2627773, -38.6191330, 38.6346664
14: -59.5215416, -2.1053715, -59.5375710, -2.0225945, -56.9666290, 56.9978790
15: -14.2561016, 18.8300724, -14.2764969, 18.8332081, -28.0895157, 28.0739517
16: -15.6869373, 22.2964249, -15.7009163, 22.3079319, -29.6646347, 29.6715736
17: -59.2392464, -6.9064512, -59.2482948, -6.8477430, -51.4531555, 51.4619598
18: -22.0583134, 16.3329105, -22.0636139, 16.3798103, -33.1265335, 33.1275482
19: -22.1702633, 6.0581493, -22.1783752, 6.1055617, -22.0151062, 22.0237656
20: -27.8465939, 0.6458912, -27.8533859, 0.6944547, -22.3877907, 22.4263649
21: -26.3223610, 7.4183102, -26.3316212, 7.4812584, -28.9014664, 28.9305115
22: -29.4883041, 5.3133378, -29.4959049, 5.3352833, -27.5309296, 27.5385475
23: -17.8375072, 12.3090210, -17.8444443, 12.3545151, -24.0536346, 24.0534134
24: -16.4992485, 13.6503906, -16.5039234, 13.6809368, -25.9053040, 25.8574448
25: -23.8380661, 8.9173851, -23.8421803, 8.9518204, -27.9853516, 27.9876709
26: -39.3408890, 4.0808959, -39.3463020, 4.1741390, -33.2894287, 33.2745819
27: -19.5091019, 14.8920946, -19.5201416, 14.9261169, -34.4352188, 34.4122353
28: -21.3899193, 11.4830856, -21.3998070, 11.5365372, -26.3398285, 26.3361702
29: -24.4068203, 6.4155302, -24.4153137, 6.4408703, -28.6392593, 28.6392555
30: -30.0183334, 5.8725338, -30.0245781, 5.9254560, -34.0057220, 33.9992676
31: -23.3651619, 7.6892734, -23.3742847, 7.7300434, -25.0975418, 25.1030426
32: -37.0009155, -2.5765281, -37.0067406, -2.5629821, -29.8589554, 29.8650894
33: -54.4473267, 0.1055431, -54.4903259, 0.1152229, -44.5081787, 44.4946442
34: -49.1645164, -6.9565134, -49.1716843, -6.9518781, -35.9251633, 35.8963165
35: -40.3152924, 3.9423923, -40.3262634, 3.9431076, -36.0691757, 36.0685349
36: -45.6247711, 1.3826017, -45.6293907, 1.3875551, -39.7744904, 39.8036118
37: -60.7337036, -9.8388023, -60.7377663, -9.8175163, -44.9564056, 44.9498444
38: -53.6782608, 4.1257992, -53.6829720, 4.1323576, -48.8978882, 48.9287567
39: -61.9477386, -4.5344086, -61.9647064, -4.5228958, -48.3489532, 48.3925629
40: -50.1927567, -9.1606569, -50.1988487, -9.1547489, -39.1399078, 39.1319962
41: -32.0095940, 7.3306594, -32.0178375, 7.3410149, -38.0462341, 38.0370026
42: -30.1150169, -0.2818432, -30.1198463, -0.2719316, -23.1470718, 23.1468391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7706147, upper bound: 13.7903738
time: 31.07 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7706147, upper bound: 13.8068501
time: 345.31 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1543045, 25.9992828, -8.0642672, 25.9209671, -30.9576340, 30.9652786
1: -0.4316406, 26.8781548, -0.3729501, 26.7820454, -21.4550323, 21.4997139
2: -0.4740982, 25.9477615, -0.3997583, 25.8055000, -20.9353218, 21.0133553
3: -4.8112569, 22.6467438, -4.7318954, 22.4813614, -19.9671211, 20.0425415
4: -7.7369857, 22.6404171, -7.6626301, 22.5101967, -25.7020912, 25.7690849
5: -4.9098892, 24.9678326, -4.8202658, 24.8145752, -23.8303871, 23.9129562
6: -39.4519234, -4.1179094, -39.4196243, -4.1456261, -29.8356323, 29.8148956
7: -9.2753029, 23.4797440, -9.1915979, 23.4195099, -25.8249245, 25.8293991
8: -13.9421682, 20.0977592, -13.8790255, 19.9804668, -29.3312454, 29.3758125
9: -8.6390553, 22.5530834, -8.6138258, 22.5103931, -27.9201431, 27.8463898
10: -29.1042519, 17.9525108, -28.9958611, 17.8657913, -39.7852707, 39.7280273
11: -26.4667187, 6.9226589, -26.1711731, 6.8680120, -29.9518890, 29.6713867
12: -46.2160912, -8.2542782, -46.0276108, -8.3910275, -32.4543533, 32.3660736
13: -32.6129341, 13.3290815, -32.6088905, 13.2296438, -38.6222076, 38.6448441
14: -59.7176933, -2.0451965, -59.4359398, -2.1515551, -57.1014862, 56.9061279
15: -14.2665920, 18.8756886, -14.2249374, 18.8126450, -28.0932159, 28.0840530
16: -15.7579651, 22.3003807, -15.6637764, 22.2795582, -29.7282486, 29.6355476
17: -59.4684105, -6.8499813, -59.1641083, -6.9157257, -51.6645432, 51.4025879
18: -22.2290554, 16.3807945, -22.0011177, 16.3484039, -33.2359543, 33.1249619
19: -22.3184929, 6.1120358, -22.1229038, 6.0736847, -22.1810303, 21.9920387
20: -27.9641571, 0.6924219, -27.8023224, 0.6374469, -22.5197372, 22.3849144
21: -26.5116463, 7.4829869, -26.2605705, 7.4197865, -29.1000290, 28.8863907
22: -29.5801754, 5.3474317, -29.4756660, 5.3122096, -27.6022720, 27.4831467
23: -17.9718933, 12.3651495, -17.7937832, 12.3221798, -24.2019577, 24.0507507
24: -16.5756226, 13.7022018, -16.4883366, 13.6872902, -25.9559174, 25.8835297
25: -23.9212875, 8.9599247, -23.8171310, 8.9277954, -28.0704269, 27.9954300
26: -39.5431595, 4.1519461, -39.2633362, 4.0567951, -33.4693451, 33.2233124
27: -19.6001282, 14.9512825, -19.4934692, 14.9383621, -34.5384903, 34.4447517
28: -21.5277710, 11.5466938, -21.3508301, 11.5024061, -26.4915466, 26.3281822
29: -24.5256882, 6.4480205, -24.3814392, 6.4154787, -28.7504044, 28.5884476
30: -30.1741066, 5.9537592, -29.9874096, 5.9014239, -34.1711273, 34.0284729
31: -23.5041618, 7.7401428, -23.3203487, 7.7105913, -25.2620697, 25.0871086
32: -37.0195389, -2.5724115, -36.9882584, -2.6209555, -29.8345184, 29.8451157
33: -54.4879303, 0.2346544, -54.4359970, 0.0577354, -44.4649506, 44.6467133
34: -49.1719742, -6.9220533, -49.1353722, -6.9844270, -35.9445572, 35.8987656
35: -40.3174820, 3.9775238, -40.2982025, 3.9252768, -36.0632477, 36.1064682
36: -45.6458740, 1.4018564, -45.6171494, 1.3580046, -39.7851257, 39.7800980
37: -60.7801895, -9.8246403, -60.7057266, -9.8607960, -44.9277954, 44.9660339
38: -53.7095985, 4.1471977, -53.6561432, 4.0918159, -48.8490601, 48.9924698
39: -61.9763641, -4.4447765, -61.9559708, -4.5598040, -48.3279266, 48.4760590
40: -50.2216911, -9.1226273, -50.1702843, -9.1724768, -39.1106415, 39.1715622
41: -32.0340271, 7.3500581, -31.9967117, 7.3212581, -38.0488281, 38.0342865
42: -30.1314087, -0.2627010, -30.1084862, -0.3022161, -23.1523819, 23.1447411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7710275
time: 28.77 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7875031
time: 36.00 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0847282, 25.9768581, -8.1224346, 25.9390411, -30.9095802, 30.9780960
1: -0.3790083, 26.8603363, -0.3992424, 26.8063316, -21.4443855, 21.5051804
2: -0.4648218, 25.9283104, -0.4859099, 25.8449249, -20.9830780, 21.0481491
3: -4.8227396, 22.6345367, -4.8382406, 22.5297928, -20.0486259, 20.0699730
4: -7.6960878, 22.6283760, -7.7278013, 22.5515785, -25.7201309, 25.7748108
5: -4.9138041, 24.9490433, -4.9334912, 24.8623886, -23.9041634, 23.9442291
6: -39.4454575, -4.1801605, -39.4354591, -4.1576834, -29.7963791, 29.7714539
7: -9.2198172, 23.4518490, -9.2410107, 23.4335804, -25.8359528, 25.8544388
8: -13.8973255, 20.0792122, -13.9279728, 20.0219612, -29.3472443, 29.3983421
9: -8.5546579, 22.5432053, -8.5950966, 22.5567245, -27.9010773, 27.8544426
10: -29.0286674, 17.9735661, -29.0042152, 17.9754162, -39.8291931, 39.7566681
11: -26.4313698, 6.9493370, -26.2317314, 6.9503465, -29.8926544, 29.7675934
12: -46.2002029, -8.2226982, -46.0995483, -8.1987505, -32.5733719, 32.4604111
13: -32.5706177, 13.3217525, -32.6024361, 13.2624359, -38.5986252, 38.6413422
14: -59.6499557, -1.9791527, -59.5076752, -1.9609356, -57.1374512, 57.0148621
15: -14.2479658, 18.8599167, -14.2577438, 18.8249073, -28.0878220, 28.0638504
16: -15.6735592, 22.2690201, -15.6634140, 22.3002567, -29.6933594, 29.6046753
17: -59.4098282, -6.8190155, -59.2217712, -6.8110123, -51.6677704, 51.4262543
18: -22.2046242, 16.3777180, -22.0584946, 16.3958092, -33.2155151, 33.1326981
19: -22.3055611, 6.1123180, -22.1781254, 6.1250982, -22.1711998, 22.0580559
20: -27.9483738, 0.6945434, -27.8549194, 0.7102904, -22.5139160, 22.4590721
21: -26.4951401, 7.5046682, -26.3295708, 7.5141010, -29.1099472, 28.9919739
22: -29.5482883, 5.3094587, -29.4942856, 5.3244519, -27.5859070, 27.5590553
23: -17.9594517, 12.3568001, -17.8452148, 12.3693638, -24.1913376, 24.0916100
24: -16.5620193, 13.6767073, -16.5029068, 13.6886368, -25.9552155, 25.8736038
25: -23.8988132, 8.9314089, -23.8416958, 8.9511623, -28.0409851, 28.0075684
26: -39.5064545, 4.1705942, -39.3447571, 4.2065678, -33.4947510, 33.3469925
27: -19.5836143, 14.9197416, -19.5181828, 14.9352570, -34.5188713, 34.4379234
28: -21.5125256, 11.5287018, -21.4030838, 11.5494061, -26.4743195, 26.3736305
29: -24.5013733, 6.4251223, -24.4124718, 6.4381828, -28.7520905, 28.6615143
30: -30.1655350, 5.9620886, -30.0231915, 5.9574599, -34.1777725, 34.0623856
31: -23.4900455, 7.7221918, -23.3742561, 7.7401142, -25.2330246, 25.1091766
32: -37.0092812, -2.5947566, -37.0063324, -2.5821981, -29.8418884, 29.8444366
33: -54.4891396, 0.1708517, -54.5176735, 0.0859852, -44.4892273, 44.6393204
34: -49.1457138, -7.0069203, -49.1729584, -6.9930730, -35.8969421, 35.8516083
35: -40.2855110, 3.8946095, -40.3237114, 3.9015961, -36.0121307, 36.0580292
36: -45.5984116, 1.3171597, -45.6277695, 1.3435631, -39.7344131, 39.7539978
37: -60.7418556, -9.8716288, -60.7344055, -9.8452682, -44.9091797, 44.9783249
38: -53.6452370, 4.0333996, -53.6796303, 4.0708551, -48.7751312, 48.9072418
39: -61.9426537, -4.4907131, -61.9671097, -4.5430641, -48.3168945, 48.4558487
40: -50.2080421, -9.1493998, -50.1945267, -9.1670084, -39.1234131, 39.1574326
41: -32.0138245, 7.2960000, -32.0187988, 7.3123226, -38.0155182, 38.0032806
42: -30.1194038, -0.3084402, -30.1209984, -0.2987080, -23.1367607, 23.1209183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7687392, upper bound: 13.8062907
time: 37.70 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7852206, upper bound: 13.8062907
time: 30.87 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.2109756, 26.0002289, -8.1855307, 25.9409523, -31.0367126, 31.0653839
1: -0.4627452, 26.8793068, -0.4411831, 26.8070984, -21.5157547, 21.5668793
2: -0.5331841, 25.9497280, -0.5207038, 25.8460217, -21.0352936, 21.1040726
3: -4.8775444, 22.6487579, -4.8663979, 22.5319023, -20.0915070, 20.1160927
4: -7.7909961, 22.6433601, -7.7752180, 22.5529175, -25.7961159, 25.8395271
5: -4.9828272, 24.9715996, -4.9690094, 24.8649731, -23.9575005, 24.0042038
6: -39.4583817, -4.1092615, -39.4395599, -4.1228390, -29.8630524, 29.8479843
7: -9.3207321, 23.4821491, -9.2925472, 23.4351883, -25.9086990, 25.9378128
8: -13.9879150, 20.1018047, -13.9744987, 20.0248222, -29.4235840, 29.4684792
9: -8.6482506, 22.5743694, -8.6410360, 22.5581779, -27.9758987, 27.9325218
10: -29.1102562, 18.0059586, -29.0457687, 17.9801064, -39.8903961, 39.8287506
11: -26.4704666, 6.9685202, -26.2510834, 6.9591475, -29.9495850, 29.8065109
12: -46.2191238, -8.1394634, -46.1030197, -8.1576443, -32.6291809, 32.5550385
13: -32.6221466, 13.3445215, -32.6280670, 13.2704554, -38.6616058, 38.7202988
14: -59.7366409, -1.9487152, -59.5501633, -1.9571896, -57.2329407, 57.1157379
15: -14.2891006, 18.8780899, -14.2762718, 18.8349361, -28.1405716, 28.1073151
16: -15.7721100, 22.3097000, -15.7114697, 22.3013763, -29.7532578, 29.6949234
17: -59.4757271, -6.7955341, -59.2543716, -6.8025484, -51.7444916, 51.5439301
18: -22.2343750, 16.4148083, -22.0670757, 16.4142380, -33.2651520, 33.1748734
19: -22.3278160, 6.1463737, -22.1844292, 6.1425061, -22.2131653, 22.0937500
20: -27.9717751, 0.7395401, -27.8585052, 0.7328439, -22.5601807, 22.5007744
21: -26.5203362, 7.5382066, -26.3382931, 7.5310760, -29.1540146, 29.0319176
22: -29.5868797, 5.3655124, -29.5011616, 5.3534493, -27.6543350, 27.5931206
23: -17.9796810, 12.3978615, -17.8496113, 12.3895683, -24.2335663, 24.1245956
24: -16.5819874, 13.7107944, -16.5071735, 13.7058105, -25.9924545, 25.9034805
25: -23.9254570, 8.9839811, -23.8450985, 8.9780560, -28.0956192, 28.0386276
26: -39.5483627, 4.2472916, -39.3500175, 4.2454605, -33.5756989, 33.4052505
27: -19.6130810, 14.9600067, -19.5281448, 14.9553146, -34.5683975, 34.4881516
28: -21.5387535, 11.5837049, -21.4073849, 11.5777864, -26.5316162, 26.4184341
29: -24.5326443, 6.4688082, -24.4210892, 6.4607792, -28.8069305, 28.6959991
30: -30.1777248, 5.9840703, -30.0281868, 5.9674392, -34.2062073, 34.0933228
31: -23.5154991, 7.7652340, -23.3813610, 7.7616901, -25.2833862, 25.1602898
32: -37.0242767, -2.5418711, -37.0105553, -2.5558481, -29.8953323, 29.9037857
33: -54.5303993, 0.2461433, -54.5235481, 0.1237049, -44.5724030, 44.6818771
34: -49.1902733, -6.9162540, -49.1762619, -6.9479690, -35.9869614, 35.9253998
35: -40.3313446, 3.9794254, -40.3280640, 3.9444904, -36.0996399, 36.1153793
36: -45.6506996, 1.4147863, -45.6325073, 1.3926611, -39.8363876, 39.8407669
37: -60.7869987, -9.8006115, -60.7404480, -9.8088150, -44.9933319, 45.0166168
38: -53.7160721, 4.1643353, -53.6863098, 4.1357098, -48.9132690, 49.0224457
39: -61.9849091, -4.4301558, -61.9773865, -4.5128460, -48.3904877, 48.5156326
40: -50.2316170, -9.1173248, -50.2030563, -9.1514139, -39.1683655, 39.2011261
41: -32.0431252, 7.3595519, -32.0235939, 7.3440142, -38.0841827, 38.0716705
42: -30.1365471, -0.2488022, -30.1235867, -0.2688885, -23.1796494, 23.1810036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7903738
time: 29.55 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8068500, upper bound: 13.8068501
time: 31.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 63.07 seconds
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7325000, upper bound: 13.8062907
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7489824, upper bound: 13.8062907
IS_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7706147, upper bound: 13.7903738
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7706147, upper bound: 13.8068501
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7710275
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7875031
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7687392, upper bound: 13.8062907
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.7852206, upper bound: 13.8062907
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.8068500, upper bound: 13.7903738
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 63.07
Output dim: 1, lower bound: -13.8068500, upper bound: 13.8068501

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.9615307, 25.9027138, -8.0567932, 25.9139709, -30.7737465, 30.8296814
1: -0.3030424, 26.7675800, -0.3583727, 26.7708664, -21.3373947, 21.3645782
2: -0.3546917, 25.8055000, -0.4288454, 25.8152733, -20.8523064, 20.8604469
3: -4.7002163, 22.4966087, -4.7744823, 22.4913673, -19.9284286, 19.9011421
4: -7.5977521, 22.5160007, -7.6767945, 22.5173187, -25.5989342, 25.6051941
5: -4.7855453, 24.8201790, -4.8669567, 24.8243198, -23.7619591, 23.7494888
6: -39.4143791, -4.2037573, -39.4286346, -4.1664362, -29.7468719, 29.7390594
7: -9.1255646, 23.3839817, -9.1946726, 23.3985062, -25.7145081, 25.7394524
8: -13.8307228, 19.9783821, -13.8881645, 19.9888382, -29.2506294, 29.2509308
9: -8.5278111, 22.5134163, -8.5779343, 22.5388012, -27.8113708, 27.8237534
10: -28.9483185, 17.8822365, -28.9840260, 17.9358635, -39.6884842, 39.6491852
11: -26.1888103, 6.8353066, -26.2009964, 6.8938241, -29.5930939, 29.6578941
12: -46.0667419, -8.3830519, -46.0720215, -8.2756252, -32.3635025, 32.3097878
13: -32.5354462, 13.2247019, -32.5888596, 13.2433710, -38.5420456, 38.5420685
14: -59.4089241, -2.1366444, -59.4418678, -2.0382767, -56.8286285, 56.8424530
15: -14.2132235, 18.8013306, -14.2491083, 18.8022118, -28.0145874, 28.0095062
16: -15.5852585, 22.2490826, -15.6437359, 22.2939205, -29.5871124, 29.5628815
17: -59.1500092, -6.9335861, -59.1689987, -6.8752127, -51.3200531, 51.2867126
18: -22.0135269, 16.2945976, -22.0261955, 16.3509216, -33.0525436, 33.0566559
19: -22.1300564, 6.0230207, -22.1377411, 6.0759487, -21.9361420, 21.9483871
20: -27.8046455, 0.6002798, -27.8141747, 0.6607790, -22.3055344, 22.3446693
21: -26.2830086, 7.3839693, -26.2961483, 7.4559674, -28.8323059, 28.8619080
22: -29.4264297, 5.2560310, -29.4432240, 5.2908211, -27.4217682, 27.4582863
23: -17.7978191, 12.2669716, -17.8018379, 12.3204536, -23.9750671, 23.9793854
24: -16.4580002, 13.6154709, -16.4598732, 13.6493568, -25.8299942, 25.7844162
25: -23.7870903, 8.8633308, -23.7923431, 8.9064350, -27.8809204, 27.9043922
26: -39.2745056, 4.0025263, -39.2935791, 4.1170282, -33.1615067, 33.1656723
27: -19.4662437, 14.8510580, -19.4847298, 14.8982372, -34.3644791, 34.3357887
28: -21.3427544, 11.4274139, -21.3555202, 11.4923801, -26.2411270, 26.2464638
29: -24.3551750, 6.3709393, -24.3675652, 6.4083977, -28.5523300, 28.5645065
30: -29.9864292, 5.8493433, -29.9825840, 5.9006314, -33.9362335, 33.9243088
31: -23.3200665, 7.6448894, -23.3301315, 7.6932602, -25.0070343, 25.0097771
32: -36.9781380, -2.6310768, -36.9869537, -2.5968537, -29.7908020, 29.7899628
33: -54.4008865, 0.0280905, -54.4734192, 0.0696487, -44.4149323, 44.4359970
34: -49.1086578, -7.0488443, -49.1454926, -7.0070581, -35.8103409, 35.7962418
35: -40.2669296, 3.8565865, -40.3159904, 3.8969250, -35.9730530, 36.0034790
36: -45.5615959, 1.2837648, -45.6035614, 1.3317890, -39.6514893, 39.6930542
37: -60.6703491, -9.9111252, -60.6953087, -9.8641491, -44.8561554, 44.8854904
38: -53.5959396, 3.9923086, -53.6532097, 4.0599995, -48.7337494, 48.7851028
39: -61.8903961, -4.5972452, -61.9247513, -4.5624456, -48.2665253, 48.3101349
40: -50.1581650, -9.1943340, -50.1688690, -9.1776962, -39.0860596, 39.0712967
41: -31.9787712, 7.2647114, -32.0097046, 7.3032618, -37.9687500, 37.9616318
42: -30.0926857, -0.3433099, -30.1068172, -0.3081450, -23.0922470, 23.0758209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7105666, upper bound: 13.8054208
time: 30.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7316378, upper bound: 13.8054208
time: 27.19 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.9668789, 25.9150352, -8.0812817, 25.9369717, -30.7779083, 30.8707733
1: -0.3042173, 26.7847900, -0.3751841, 26.8036671, -21.3396416, 21.4050217
2: -0.3566842, 25.8188839, -0.4434981, 25.8413086, -20.8508148, 20.8971939
3: -4.7013268, 22.5152397, -4.7895684, 22.5269318, -19.9221001, 19.9478836
4: -7.5992627, 22.5305977, -7.6912851, 22.5465279, -25.6095505, 25.6404762
5: -4.7876539, 24.8384094, -4.8838043, 24.8591862, -23.7593880, 23.7941971
6: -39.4157257, -4.2011185, -39.4304161, -4.1585970, -29.7566376, 29.7449341
7: -9.1277781, 23.4009933, -9.2114925, 23.4308491, -25.7196198, 25.7777557
8: -13.8318949, 19.9930420, -13.9041748, 20.0168076, -29.2511826, 29.2861443
9: -8.5299864, 22.5205135, -8.5882893, 22.5516052, -27.8172760, 27.8438797
10: -28.9552059, 17.8862419, -28.9986153, 17.9493484, -39.7055435, 39.6576385
11: -26.2014389, 6.8364744, -26.2254238, 6.9030495, -29.6156464, 29.6712189
12: -46.0782890, -8.3800106, -46.0946426, -8.2607594, -32.3923187, 32.3129425
13: -32.5375175, 13.2297306, -32.5952644, 13.2512913, -38.5486679, 38.5541229
14: -59.4334145, -2.1358480, -59.4920464, -2.0266266, -56.8699341, 56.8646851
15: -14.2147446, 18.8105259, -14.2576189, 18.8200493, -28.0206985, 28.0286636
16: -15.5880890, 22.2553959, -15.6520405, 22.3059578, -29.5945358, 29.5796356
17: -59.1722641, -6.9302540, -59.2128601, -6.8569698, -51.3752136, 51.2913208
18: -22.0275059, 16.2957153, -22.0526085, 16.3611641, -33.0760803, 33.0674591
19: -22.1471214, 6.0239782, -22.1700249, 6.0879555, -21.9727020, 21.9479103
20: -27.8221283, 0.6008620, -27.8475494, 0.6717558, -22.3409195, 22.3457909
21: -26.2960510, 7.3846755, -26.3202934, 7.4641566, -28.8565826, 28.8671722
22: -29.4486370, 5.2571173, -29.4865799, 5.3058538, -27.4612732, 27.4663315
23: -17.8164463, 12.2678604, -17.8380604, 12.3340893, -24.0104523, 23.9905281
24: -16.4782848, 13.6162758, -16.4973335, 13.6636696, -25.8672333, 25.7986755
25: -23.8105583, 8.8646774, -23.8364868, 8.9245300, -27.9297791, 27.9053802
26: -39.2980881, 4.0041084, -39.3385849, 4.1349812, -33.2075043, 33.1757889
27: -19.4786186, 14.8517847, -19.5079269, 14.9059467, -34.3845673, 34.3597107
28: -21.3628807, 11.4280100, -21.3935051, 11.5079231, -26.2820358, 26.2517662
29: -24.3744125, 6.3717647, -24.4042015, 6.4180784, -28.5830307, 28.5848351
30: -30.0052319, 5.8504219, -30.0173721, 5.9151368, -33.9764709, 33.9345322
31: -23.3388252, 7.6461062, -23.3650837, 7.7082930, -25.0466690, 25.0106354
32: -36.9852180, -2.6295605, -37.0009003, -2.5896382, -29.8039703, 29.8017883
33: -54.4055138, 0.0300522, -54.4829865, 0.0768747, -44.4186554, 44.4572220
34: -49.1196060, -7.0474401, -49.1673203, -6.9975314, -35.8342361, 35.8134689
35: -40.2691917, 3.8573980, -40.3211899, 3.8997669, -35.9815979, 36.0100555
36: -45.5719070, 1.2847919, -45.6233978, 1.3379936, -39.6712799, 39.7112732
37: -60.6868057, -9.9098816, -60.7274704, -9.8541813, -44.8661194, 44.9083023
38: -53.6062965, 3.9947157, -53.6736526, 4.0668144, -48.7581635, 48.8013153
39: -61.9046860, -4.5953426, -61.9526062, -4.5537128, -48.2704315, 48.3348236
40: -50.1682587, -9.1929131, -50.1879425, -9.1707745, -39.0911407, 39.0902328
41: -31.9802036, 7.2669592, -32.0127182, 7.3089628, -37.9764252, 37.9676666
42: -30.0971909, -0.3416247, -30.1156826, -0.3020878, -23.1030006, 23.0803871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7270532, upper bound: 13.8054208
time: 31.74 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7481174, upper bound: 13.8054208
time: 37.03 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.0925102, 25.9376488, -8.1449909, 25.9396076, -30.9318733, 30.9312286
1: -0.3877382, 26.8027992, -0.4173107, 26.8053684, -21.4477577, 21.4299965
2: -0.4248729, 25.8394794, -0.4785452, 25.8431644, -20.9380188, 20.9181023
3: -4.7559752, 22.5282269, -4.8177953, 22.5302353, -20.0099564, 19.9490051
4: -7.6938820, 22.5443115, -7.7389460, 22.5491714, -25.7103233, 25.6804199
5: -4.8563805, 24.8598366, -4.9195299, 24.8628273, -23.8592377, 23.8075333
6: -39.4283180, -4.1304693, -39.4348373, -4.1234789, -29.8230820, 29.8216934
7: -9.2283478, 23.4303818, -9.2633762, 23.4333344, -25.8241997, 25.8292618
8: -13.9223156, 20.0147057, -13.9509287, 20.0206280, -29.3561707, 29.3276176
9: -8.6233673, 22.5503807, -8.6344786, 22.5543308, -27.9032745, 27.9109001
10: -29.0357323, 17.9181576, -29.0412655, 17.9545021, -39.7621994, 39.7343216
11: -26.2388115, 6.8555183, -26.2464447, 6.9119811, -29.6585541, 29.7242203
12: -46.0954285, -8.2970877, -46.0998535, -8.2193794, -32.4310760, 32.4246368
13: -32.5887985, 13.2505932, -32.6210594, 13.2613029, -38.6175690, 38.6271667
14: -59.5185623, -2.1056843, -59.5361633, -2.0226707, -56.9342499, 56.9967194
15: -14.2556791, 18.8269424, -14.2762985, 18.8318501, -28.0877228, 28.0579262
16: -15.6861668, 22.2955914, -15.7006054, 22.3075409, -29.6629486, 29.6613388
17: -59.2364388, -6.9071732, -59.2470474, -6.8480930, -51.4002914, 51.4606628
18: -22.0558186, 16.3327560, -22.0626183, 16.3797264, -33.1086273, 33.1266174
19: -22.1682415, 6.0579929, -22.1774979, 6.1054897, -21.9749374, 22.0233269
20: -27.8443298, 0.6457438, -27.8522987, 0.6943984, -22.3489456, 22.4258080
21: -26.3197708, 7.4181480, -26.3305340, 7.4811792, -28.8780975, 28.9297256
22: -29.4857769, 5.3129630, -29.4948502, 5.3351398, -27.4927673, 27.5373878
23: -17.8355293, 12.3088026, -17.8435993, 12.3543968, -24.0237350, 24.0524406
24: -16.4969101, 13.6502686, -16.5029335, 13.6808662, -25.8763580, 25.8564835
25: -23.8357964, 8.9169998, -23.8412552, 8.9516468, -27.9341507, 27.9867096
26: -39.3385124, 4.0806699, -39.3453064, 4.1740284, -33.2489090, 33.2736130
27: -19.5068092, 14.8919582, -19.5191307, 14.9260540, -34.4328613, 34.4110870
28: -21.3879318, 11.4829388, -21.3990211, 11.5364437, -26.3002777, 26.3357124
29: -24.4042969, 6.4152703, -24.4141808, 6.4407587, -28.6193314, 28.6378975
30: -30.0161591, 5.8722019, -30.0236588, 5.9253149, -33.9719086, 33.9984512
31: -23.3630409, 7.6890373, -23.3734245, 7.7299743, -25.0562668, 25.1025467
32: -36.9993668, -2.5768628, -37.0060463, -2.5631123, -29.8549728, 29.8635788
33: -54.4458466, 0.1050110, -54.4897652, 0.1149731, -44.5134125, 44.4882812
34: -49.1634979, -6.9570432, -49.1712036, -6.9520903, -35.9160690, 35.8954697
35: -40.3145866, 3.9419203, -40.3259697, 3.9429417, -36.0680847, 36.0685120
36: -45.6234550, 1.3820858, -45.6288719, 1.3873558, -39.7689056, 39.8024368
37: -60.7294693, -9.8390446, -60.7360001, -9.8176374, -44.9532013, 44.9435654
38: -53.6756668, 4.1251535, -53.6818314, 4.1320448, -48.8856201, 48.9271927
39: -61.9459343, -4.5349846, -61.9639397, -4.5232029, -48.3510284, 48.3876572
40: -50.1903534, -9.1610603, -50.1978149, -9.1549492, -39.1418304, 39.1281357
41: -32.0093155, 7.3303218, -32.0177155, 7.3408813, -38.0452881, 38.0358047
42: -30.1134262, -0.2821927, -30.1192055, -0.2720828, -23.1407242, 23.1456451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7697505, upper bound: 13.7849337
time: 27.34 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7697505, upper bound: 13.8059852
time: 30.42 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.1286564, 25.9750538, -8.0585022, 25.9081230, -30.9151382, 30.9329376
1: -0.4144268, 26.8437901, -0.3716135, 26.7642097, -21.4140625, 21.4602318
2: -0.4589834, 25.9201775, -0.3976202, 25.7913704, -20.8979797, 20.9792747
3: -4.7959819, 22.6089458, -4.7307038, 22.4616852, -19.9196548, 20.0031624
4: -7.7220893, 22.6086845, -7.6609254, 22.4943314, -25.6661072, 25.7329750
5: -4.8926439, 24.9310970, -4.8180246, 24.7955894, -23.7850037, 23.8683090
6: -39.4496117, -4.1262589, -39.4180107, -4.1485310, -29.8285599, 29.8041458
7: -9.2578754, 23.4458046, -9.1890812, 23.4018345, -25.7855988, 25.7914009
8: -13.9258671, 20.0682240, -13.8777494, 19.9651871, -29.2952385, 29.3458023
9: -8.6283102, 22.5381165, -8.6114616, 22.5024529, -27.8991318, 27.8285446
10: -29.0877571, 17.9382477, -28.9880562, 17.8614216, -39.7693863, 39.7080231
11: -26.4392567, 6.9132032, -26.1571884, 6.8667831, -29.9232330, 29.6475182
12: -46.1905174, -8.2696991, -46.0148773, -8.3942766, -32.4308853, 32.3339233
13: -32.6062012, 13.3177252, -32.6066856, 13.2231598, -38.6084747, 38.6307983
14: -59.6645508, -2.0571318, -59.4100990, -2.1523972, -57.0468750, 56.8636932
15: -14.2577248, 18.8546753, -14.2232428, 18.8019600, -28.0722427, 28.0619202
16: -15.7489643, 22.2875061, -15.6605701, 22.2729359, -29.7098236, 29.6179352
17: -59.4218216, -6.8690052, -59.1406860, -6.9193459, -51.6069031, 51.3461304
18: -22.2001362, 16.3703537, -21.9860764, 16.3472404, -33.2072296, 33.1005478
19: -22.2841949, 6.0998049, -22.1049519, 6.0726509, -22.1413498, 21.9550171
20: -27.9285603, 0.6813068, -27.7837639, 0.6367950, -22.4797783, 22.3489304
21: -26.4849377, 7.4746671, -26.2465019, 7.4190555, -29.0714264, 28.8612976
22: -29.5343189, 5.3320570, -29.4524059, 5.3109827, -27.5560608, 27.4424133
23: -17.9336815, 12.3512726, -17.7743473, 12.3211432, -24.1609039, 24.0144081
24: -16.5357819, 13.6877480, -16.4670448, 13.6864214, -25.9127045, 25.8453751
25: -23.8748283, 8.9414215, -23.7927170, 8.9263725, -28.0181808, 27.9456444
26: -39.4958076, 4.1337643, -39.2387733, 4.0551519, -33.4186554, 33.1764221
27: -19.5746269, 14.9434891, -19.4801216, 14.9375839, -34.5122108, 34.4236107
28: -21.4877396, 11.5309792, -21.3298798, 11.5017042, -26.4466934, 26.2867889
29: -24.4865761, 6.4381266, -24.3611755, 6.4145861, -28.7101517, 28.5564613
30: -30.1370907, 5.9389100, -29.9676533, 5.9001589, -34.1270676, 33.9874496
31: -23.4670448, 7.7249117, -23.3006973, 7.7093143, -25.2199478, 25.0469704
32: -37.0040092, -2.5799313, -36.9804916, -2.6226716, -29.8187027, 29.8304367
33: -54.4768639, 0.2268877, -54.4307747, 0.0556068, -44.4490509, 44.6366653
34: -49.1491394, -6.9322186, -49.1239586, -6.9861007, -35.9182205, 35.8740158
35: -40.3116074, 3.9742575, -40.2956619, 3.9242306, -36.0555115, 36.0978928
36: -45.6247711, 1.3952570, -45.6062202, 1.3568211, -39.7613525, 39.7591400
37: -60.7437668, -9.8347664, -60.6874924, -9.8621655, -44.9016876, 44.9498520
38: -53.6866188, 4.1397200, -53.6446381, 4.0891953, -48.8205719, 48.9664536
39: -61.9466438, -4.4541264, -61.9408531, -4.5620117, -48.3053436, 48.4672928
40: -50.2002335, -9.1299667, -50.1591949, -9.1740379, -39.0935822, 39.1627121
41: -32.0306816, 7.3439436, -31.9951248, 7.3188367, -38.0419006, 38.0254822
42: -30.1209297, -0.2690477, -30.1033287, -0.3040676, -23.1414452, 23.1328087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7849336, upper bound: 13.7701640
time: 29.38 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7701640
time: 31.20 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.1531849, 25.9980469, -8.0637732, 25.9204655, -30.9562187, 30.9371033
1: -0.4312315, 26.8766117, -0.3727927, 26.7814140, -21.4545250, 21.4624786
2: -0.4736991, 25.9462223, -0.3995857, 25.8047447, -20.9347153, 20.9777832
3: -4.8110461, 22.6444740, -4.7318254, 22.4803181, -19.9663925, 19.9968529
4: -7.7365685, 22.6379089, -7.6624784, 22.5089569, -25.7014008, 25.7435799
5: -4.9094968, 24.9659958, -4.8201075, 24.8138313, -23.8297310, 23.8657379
6: -39.4513893, -4.1183825, -39.4193802, -4.1458273, -29.8343964, 29.8139114
7: -9.2746849, 23.4781494, -9.1913509, 23.4188919, -25.8239174, 25.7965393
8: -13.9418755, 20.0962181, -13.8788853, 19.9798584, -29.3304443, 29.3463707
9: -8.6385946, 22.5509262, -8.6136742, 22.5095501, -27.9192810, 27.8344574
10: -29.1023235, 17.9517136, -28.9950466, 17.8654251, -39.7778091, 39.7251282
11: -26.4636555, 6.9224820, -26.1698093, 6.8679485, -29.9365768, 29.6700974
12: -46.2131157, -8.2547436, -46.0263901, -8.3911934, -32.4340210, 32.3627930
13: -32.6125717, 13.3256817, -32.6087265, 13.2282009, -38.6205902, 38.6373978
14: -59.7146301, -2.0454817, -59.4345398, -2.1516151, -57.0690918, 56.9049988
15: -14.2662086, 18.8725224, -14.2247686, 18.8112392, -28.0913849, 28.0680275
16: -15.7572021, 22.2995434, -15.6633606, 22.2792473, -29.7265778, 29.6253586
17: -59.4656219, -6.8507242, -59.1628799, -6.9160404, -51.6116486, 51.4012451
18: -22.2265701, 16.3805923, -22.0000839, 16.3483162, -33.2180252, 33.1240387
19: -22.3164463, 6.1118374, -22.1220245, 6.0735912, -22.1408844, 21.9916115
20: -27.9619293, 0.6922812, -27.8012371, 0.6373858, -22.4808846, 22.3843498
21: -26.5090485, 7.4828291, -26.2595177, 7.4197350, -29.0766525, 28.8855591
22: -29.5776825, 5.3470869, -29.4746132, 5.3120661, -27.5641174, 27.4819756
23: -17.9698925, 12.3649197, -17.7929382, 12.3220615, -24.1720581, 24.0498047
24: -16.5732651, 13.7020741, -16.4873638, 13.6872244, -25.9269562, 25.8825836
25: -23.9190464, 8.9594812, -23.8161697, 8.9276218, -28.0192108, 27.9944763
26: -39.5407181, 4.1516895, -39.2623291, 4.0567136, -33.4287643, 33.2223740
27: -19.5978699, 14.9511499, -19.4924908, 14.9383297, -34.5362015, 34.4436417
28: -21.5257568, 11.5465031, -21.3499985, 11.5022984, -26.4520111, 26.3277168
29: -24.5231972, 6.4477944, -24.3803635, 6.4154272, -28.7304611, 28.5871429
30: -30.1719246, 5.9534583, -29.9865093, 5.9012871, -34.1373444, 34.0276718
31: -23.5020237, 7.7399416, -23.3195133, 7.7105303, -25.2207947, 25.0865898
32: -37.0179863, -2.5727649, -36.9875793, -2.6210847, -29.8305130, 29.8435974
33: -54.4864883, 0.2340870, -54.4353943, 0.0575161, -44.4702148, 44.6403503
34: -49.1709175, -6.9225945, -49.1349716, -6.9846363, -35.9354858, 35.8978958
35: -40.3167953, 3.9770842, -40.2979088, 3.9251423, -36.0621490, 36.1064453
36: -45.6445770, 1.4013863, -45.6166000, 1.3577948, -39.7794876, 39.7789078
37: -60.7759628, -9.8248863, -60.7039604, -9.8608494, -44.9245911, 44.9598160
38: -53.7070236, 4.1465607, -53.6550598, 4.0915699, -48.8367920, 48.9908752
39: -61.9745255, -4.4453697, -61.9551620, -4.5600624, -48.3299713, 48.4711151
40: -50.2193146, -9.1230173, -50.1693115, -9.1726694, -39.1125183, 39.1677399
41: -32.0336761, 7.3497353, -31.9965687, 7.3210626, -38.0479126, 38.0331421
42: -30.1298275, -0.2630281, -30.1078396, -0.3023767, -23.1460114, 23.1435432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7849336, upper bound: 13.7866375
time: 35.04 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7866375
time: 27.82 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0789337, 25.9640312, -8.0968361, 25.9148178, -30.8772125, 30.9356079
1: -0.3776722, 26.8424950, -0.3820558, 26.7719574, -21.4049034, 21.4641991
2: -0.4626555, 25.9141922, -0.4708395, 25.8173580, -20.9489899, 21.0108452
3: -4.8215799, 22.6148796, -4.8229880, 22.4919930, -20.0092545, 20.0224915
4: -7.6944041, 22.6125145, -7.7129197, 22.5198383, -25.6839943, 25.7387695
5: -4.9115806, 24.9300785, -4.9162693, 24.8256969, -23.8595505, 23.8988647
6: -39.4438782, -4.1829929, -39.4331589, -4.1660004, -29.7855911, 29.7643661
7: -9.2173452, 23.4341545, -9.2235546, 23.3996944, -25.7979660, 25.8151169
8: -13.8960686, 20.0639229, -13.9116211, 19.9924145, -29.3172379, 29.3623352
9: -8.5523119, 22.5352669, -8.5843086, 22.5417480, -27.8832321, 27.8334618
10: -29.0209522, 17.9692993, -28.9877014, 17.9611206, -39.8092041, 39.7407684
11: -26.4173450, 6.9480944, -26.2042923, 6.9409204, -29.8687668, 29.7389374
12: -46.1875076, -8.2259722, -46.0739212, -8.2140932, -32.5412369, 32.4369049
13: -32.5684128, 13.3151875, -32.5957413, 13.2510757, -38.5845566, 38.6276550
14: -59.6241417, -1.9799976, -59.4545631, -1.9728699, -57.0949402, 56.9603424
15: -14.2462692, 18.8492813, -14.2488461, 18.8039227, -28.0656815, 28.0428734
16: -15.6703835, 22.2623405, -15.6543598, 22.2873878, -29.6757507, 29.5862503
17: -59.3863373, -6.8226376, -59.1750908, -6.8300371, -51.6113281, 51.3686066
18: -22.1895752, 16.3765373, -22.0296421, 16.3853703, -33.1910782, 33.1040115
19: -22.2876301, 6.1112776, -22.1437950, 6.1129007, -22.1342049, 22.0183640
20: -27.9298344, 0.6938939, -27.8192940, 0.6991515, -22.4779587, 22.4190865
21: -26.4810333, 7.5038967, -26.3028755, 7.5058470, -29.0848923, 28.9633713
22: -29.5250053, 5.3082724, -29.4484825, 5.3090577, -27.5451889, 27.5128479
23: -17.9399986, 12.3557930, -17.8070011, 12.3554878, -24.1550293, 24.0505486
24: -16.5407734, 13.6758518, -16.4630890, 13.6742182, -25.9170609, 25.8304672
25: -23.8744183, 8.9299545, -23.7952576, 8.9326706, -27.9911957, 27.9553413
26: -39.4819221, 4.1688795, -39.2973480, 4.1883655, -33.4478149, 33.2963638
27: -19.5702629, 14.9189606, -19.4927597, 14.9274282, -34.4976921, 34.4117203
28: -21.4916077, 11.5280132, -21.3630104, 11.5336266, -26.4329300, 26.3287354
29: -24.4810696, 6.4242334, -24.3733215, 6.4282579, -28.7200775, 28.6212692
30: -30.1458054, 5.9608340, -29.9861832, 5.9425774, -34.1367569, 34.0183487
31: -23.4704113, 7.7208557, -23.3371735, 7.7248783, -25.1929245, 25.0670586
32: -37.0015297, -2.5963945, -36.9907951, -2.5897007, -29.8271790, 29.8286057
33: -54.4839020, 0.1686420, -54.5065842, 0.0782652, -44.4791565, 44.6233521
34: -49.1343575, -7.0086203, -49.1501045, -7.0031247, -35.8721771, 35.8253021
35: -40.2829971, 3.8936224, -40.3178406, 3.8982525, -36.0034943, 36.0503235
36: -45.5875626, 1.3160305, -45.6066971, 1.3369198, -39.7133942, 39.7302246
37: -60.7236023, -9.8730030, -60.6979370, -9.8554230, -44.8929596, 44.9523468
38: -53.6336479, 4.0307856, -53.6565895, 4.0634155, -48.7490540, 48.8787537
39: -61.9275398, -4.4929476, -61.9373970, -4.5523968, -48.3080902, 48.4332733
40: -50.1969910, -9.1509438, -50.1730957, -9.1743336, -39.1145172, 39.1404343
41: -32.0122414, 7.2935886, -32.0154724, 7.3062401, -38.0066910, 37.9962540
42: -30.1142406, -0.3102198, -30.1105404, -0.3050704, -23.1248360, 23.1099663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7468073, upper bound: 13.8054208
time: 29.89 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7678765, upper bound: 13.8054208
time: 38.36 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.0842419, 25.9763565, -8.1213236, 25.9377995, -30.8813667, 30.9767075
1: -0.3787956, 26.8597260, -0.3988581, 26.8047962, -21.4071655, 21.5046349
2: -0.4646468, 25.9275646, -0.4855013, 25.8434067, -20.9475479, 21.0475922
3: -4.8227024, 22.6334991, -4.8380423, 22.5275421, -20.0029411, 20.0692596
4: -7.6959519, 22.6271553, -7.7273750, 22.5490589, -25.6946106, 25.7740479
5: -4.9136763, 24.9482994, -4.9330812, 24.8605881, -23.8569565, 23.9435883
6: -39.4452477, -4.1803503, -39.4348793, -4.1581345, -29.7953720, 29.7702332
7: -9.2195702, 23.4512043, -9.2403984, 23.4320183, -25.8030968, 25.8534355
8: -13.8972054, 20.0785828, -13.9275904, 20.0203972, -29.3178444, 29.3975372
9: -8.5544548, 22.5423450, -8.5946760, 22.5545578, -27.8891525, 27.8535919
10: -29.0278358, 17.9732475, -29.0023003, 17.9746475, -39.8263168, 39.7491608
11: -26.4299603, 6.9492350, -26.2287331, 6.9501510, -29.8913345, 29.7522430
12: -46.1989975, -8.2228661, -46.0965843, -8.1991901, -32.5700760, 32.4400940
13: -32.5705299, 13.3202763, -32.6021042, 13.2590170, -38.5911560, 38.6397476
14: -59.6485786, -1.9792738, -59.5046387, -1.9612398, -57.1362457, 56.9825287
15: -14.2477942, 18.8585091, -14.2573376, 18.8217640, -28.0717926, 28.0620537
16: -15.6732216, 22.2686596, -15.6625900, 22.2994232, -29.6831589, 29.6030121
17: -59.4085693, -6.8192730, -59.2189789, -6.8116989, -51.6664963, 51.3733215
18: -22.2035751, 16.3776169, -22.0560436, 16.3955936, -33.2146225, 33.1147919
19: -22.3046818, 6.1122313, -22.1760883, 6.1249065, -22.1707726, 22.0178986
20: -27.9472923, 0.6944809, -27.8526688, 0.7101407, -22.5133514, 22.4202003
21: -26.4940491, 7.5046067, -26.3269997, 7.5139971, -29.1091461, 28.9686050
22: -29.5472603, 5.3092828, -29.4918633, 5.3240919, -27.5847092, 27.5209045
23: -17.9585800, 12.3566742, -17.8432312, 12.3691444, -24.1904068, 24.0617142
24: -16.5610275, 13.6766796, -16.5005665, 13.6885366, -25.9543076, 25.8446579
25: -23.8979187, 8.9312029, -23.8394070, 8.9507875, -28.0400314, 27.9563332
26: -39.5054893, 4.1705413, -39.3423309, 4.2063150, -33.4937744, 33.3064651
27: -19.5826225, 14.9196949, -19.5159435, 14.9350891, -34.5177116, 34.4356384
28: -21.5117264, 11.5286379, -21.4010773, 11.5492430, -26.4738617, 26.3340874
29: -24.5002937, 6.4250164, -24.4099884, 6.4379306, -28.7507172, 28.6416397
30: -30.1646385, 5.9619637, -30.0209999, 5.9571233, -34.1769714, 34.0285873
31: -23.4891605, 7.7220845, -23.3721294, 7.7399006, -25.2325134, 25.0679131
32: -37.0086136, -2.5948858, -37.0047913, -2.5825076, -29.8403473, 29.8404083
33: -54.4885750, 0.1706495, -54.5162430, 0.0854683, -44.4828491, 44.6445999
34: -49.1452789, -7.0071487, -49.1718864, -6.9936075, -35.8961105, 35.8425140
35: -40.2852325, 3.8944750, -40.3229904, 3.9011574, -36.0120850, 36.0569153
36: -45.5978966, 1.3169756, -45.6264648, 1.3430643, -39.7332458, 39.7484360
37: -60.7400970, -9.8717308, -60.7301598, -9.8455601, -44.9029541, 44.9751129
38: -53.6441422, 4.0331926, -53.6770401, 4.0702362, -48.7735748, 48.8949738
39: -61.9418259, -4.4909401, -61.9653015, -4.5436487, -48.3119965, 48.4579315
40: -50.2070389, -9.1495628, -50.1921387, -9.1674280, -39.1195831, 39.1593399
41: -32.0136948, 7.2958770, -32.0184822, 7.3119869, -38.0143585, 38.0022888
42: -30.1187572, -0.3085575, -30.1194134, -0.2990360, -23.1355743, 23.1145401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7632926, upper bound: 13.8054208
time: 31.90 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7843539, upper bound: 13.8054208
time: 29.31 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.1853657, 25.9759941, -8.1797085, 25.9281540, -30.9942551, 31.0330276
1: -0.4455571, 26.8449173, -0.4398208, 26.7892323, -21.4748001, 21.5273972
2: -0.5181370, 25.9221439, -0.5185499, 25.8318825, -20.9980011, 21.0699768
3: -4.8622522, 22.6109352, -4.8652134, 22.5121994, -20.0440407, 20.0767250
4: -7.7761154, 22.6116333, -7.7735257, 22.5370617, -25.7601128, 25.8033791
5: -4.9655991, 24.9348354, -4.9667578, 24.8459854, -23.9121361, 23.9595184
6: -39.4560547, -4.1175547, -39.4379654, -4.1256914, -29.8560028, 29.8372726
7: -9.3032875, 23.4482460, -9.2900639, 23.4175110, -25.8693810, 25.8998260
8: -13.9716253, 20.0722847, -13.9732304, 20.0095825, -29.3875656, 29.4384232
9: -8.6374960, 22.5593796, -8.6386862, 22.5502167, -27.9549255, 27.9146805
10: -29.0937233, 17.9917221, -29.0379982, 17.9758015, -39.8744354, 39.8087692
11: -26.4429779, 6.9590759, -26.2370777, 6.9579616, -29.9209595, 29.7826614
12: -46.1935844, -8.1548328, -46.0903015, -8.1608677, -32.6056824, 32.5229034
13: -32.6154327, 13.3331795, -32.6258202, 13.2639084, -38.6479416, 38.7061996
14: -59.6835747, -1.9606686, -59.5243301, -1.9580317, -57.1783600, 57.0733032
15: -14.2802486, 18.8571072, -14.2745771, 18.8243256, -28.1196060, 28.0851364
16: -15.7630348, 22.2968121, -15.7083473, 22.2946930, -29.7348328, 29.6773415
17: -59.4290695, -6.8145599, -59.2308884, -6.8061543, -51.6869431, 51.4875183
18: -22.2054634, 16.4043598, -22.0520058, 16.4130135, -33.2364273, 33.1503906
19: -22.2935123, 6.1341944, -22.1664848, 6.1414804, -22.1734886, 22.0567436
20: -27.9361343, 0.7284279, -27.8399315, 0.7321663, -22.5202179, 22.4648209
21: -26.4936066, 7.5298843, -26.3242397, 7.5303249, -29.1253891, 29.0068893
22: -29.5410137, 5.3501062, -29.4778976, 5.3522830, -27.6081161, 27.5523834
23: -17.9414539, 12.3839855, -17.8302059, 12.3885670, -24.1925049, 24.0882721
24: -16.5421410, 13.6963100, -16.4858379, 13.7049122, -25.9492645, 25.8652802
25: -23.8789825, 8.9655075, -23.8206787, 8.9766207, -28.0434113, 27.9888306
26: -39.5009689, 4.2291822, -39.3254395, 4.2438154, -33.5251007, 33.3583450
27: -19.5875740, 14.9522018, -19.5148315, 14.9545183, -34.5420914, 34.4670334
28: -21.4987240, 11.5679770, -21.3864098, 11.5770969, -26.4867783, 26.3770447
29: -24.4935684, 6.4588900, -24.4007759, 6.4598713, -28.7666779, 28.6640015
30: -30.1407051, 5.9692097, -30.0084267, 5.9661512, -34.1621704, 34.0523224
31: -23.4784069, 7.7500038, -23.3616886, 7.7603188, -25.2412491, 25.1201935
32: -37.0087624, -2.5493689, -37.0028381, -2.5575361, -29.8795090, 29.8891220
33: -54.5192986, 0.2384291, -54.5183296, 0.1215630, -44.5565186, 44.6718445
34: -49.1674080, -6.9263268, -49.1648483, -6.9496088, -35.9606476, 35.9006653
35: -40.3254814, 3.9761295, -40.3255348, 3.9434624, -36.0919571, 36.1068192
36: -45.6296463, 1.4081602, -45.6216164, 1.3915834, -39.8126373, 39.8197708
37: -60.7505798, -9.8108730, -60.7222290, -9.8102160, -44.9672089, 45.0003967
38: -53.6929741, 4.1568222, -53.6748352, 4.1330318, -48.8848267, 48.9964218
39: -61.9551506, -4.4394541, -61.9622955, -4.5150480, -48.3679962, 48.5068970
40: -50.2101288, -9.1246109, -50.1919632, -9.1529884, -39.1513519, 39.1921997
41: -32.0397987, 7.3534827, -32.0220146, 7.3415723, -38.0771790, 38.0628357
42: -30.1260910, -0.2551756, -30.1183968, -0.2706957, -23.1687164, 23.1690674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7684516
time: 28.41 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7895105
time: 31.28 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.2098551, 25.9989796, -8.1850176, 25.9404449, -31.0353012, 31.0371475
1: -0.4623594, 26.8777390, -0.4409571, 26.8064651, -21.5152206, 21.5296364
2: -0.5328383, 25.9481888, -0.5205441, 25.8452759, -21.0346985, 21.0684929
3: -4.8773470, 22.6465054, -4.8662877, 22.5308514, -20.0907745, 20.0704041
4: -7.7905464, 22.6408463, -7.7750349, 22.5517006, -25.7954102, 25.8139915
5: -4.9824257, 24.9697800, -4.9688692, 24.8642063, -23.9568443, 23.9569473
6: -39.4578743, -4.1097507, -39.4393120, -4.1230297, -29.8618393, 29.8470154
7: -9.3201113, 23.4805794, -9.2922850, 23.4345207, -25.9076843, 25.9049606
8: -13.9876089, 20.1002769, -13.9743729, 20.0242233, -29.4227905, 29.4390221
9: -8.6477871, 22.5722122, -8.6408806, 22.5572834, -27.9750519, 27.9206009
10: -29.1083679, 18.0051956, -29.0449486, 17.9797897, -39.8828812, 39.8258972
11: -26.4673595, 6.9682961, -26.2497234, 6.9590578, -29.9342499, 29.8052139
12: -46.2162170, -8.1398973, -46.1018219, -8.1578026, -32.6088486, 32.5517807
13: -32.6217804, 13.3411236, -32.6279335, 13.2689657, -38.6600342, 38.7128143
14: -59.7337036, -1.9490213, -59.5487518, -1.9572773, -57.2005615, 57.1145325
15: -14.2887440, 18.8749542, -14.2761116, 18.8335419, -28.1387634, 28.0912819
16: -15.7712746, 22.3088512, -15.7111473, 22.3010044, -29.7515717, 29.6847000
17: -59.4729156, -6.7962275, -59.2531052, -6.8028460, -51.6915741, 51.5426483
18: -22.2318954, 16.4146633, -22.0660248, 16.4141350, -33.2472534, 33.1739426
19: -22.3257904, 6.1462259, -22.1835670, 6.1424422, -22.1730080, 22.0933228
20: -27.9694901, 0.7393808, -27.8574486, 0.7327585, -22.5213318, 22.5002060
21: -26.5177574, 7.5380678, -26.3372250, 7.5310426, -29.1306610, 29.0311584
22: -29.5844116, 5.3651652, -29.5001431, 5.3533196, -27.6161880, 27.5919609
23: -17.9776917, 12.3976383, -17.8487873, 12.3894787, -24.2036514, 24.1236305
24: -16.5796661, 13.7106609, -16.5061684, 13.7057362, -25.9634857, 25.9025421
25: -23.9231663, 8.9836044, -23.8441734, 8.9779167, -28.0444107, 28.0376778
26: -39.5459251, 4.2470655, -39.3490295, 4.2453294, -33.5351486, 33.4043045
27: -19.6108170, 14.9598637, -19.5271645, 14.9552345, -34.5660515, 34.4870300
28: -21.5367603, 11.5835266, -21.4065552, 11.5776749, -26.4920807, 26.4179916
29: -24.5301857, 6.4685521, -24.4199905, 6.4607048, -28.7870178, 28.6946487
30: -30.1755333, 5.9837427, -30.0272388, 5.9673114, -34.1723938, 34.0925140
31: -23.5133553, 7.7650228, -23.3804646, 7.7615814, -25.2421112, 25.1597862
32: -37.0227470, -2.5422277, -37.0099106, -2.5560107, -29.8913345, 29.9022522
33: -54.5288849, 0.2456369, -54.5230026, 0.1234703, -44.5776825, 44.6755371
34: -49.1892319, -6.9167781, -49.1758270, -6.9481769, -35.9778748, 35.9245148
35: -40.3306808, 3.9790154, -40.3277702, 3.9442949, -36.0985413, 36.1153717
36: -45.6494370, 1.4143324, -45.6319466, 1.3924952, -39.8308182, 39.8395767
37: -60.7827530, -9.8008890, -60.7386551, -9.8088989, -44.9900818, 45.0103607
38: -53.7134323, 4.1636629, -53.6851997, 4.1354485, -48.9010315, 49.0208588
39: -61.9830704, -4.4307156, -61.9765968, -4.5130844, -48.3926392, 48.5106964
40: -50.2292023, -9.1176710, -50.2020531, -9.1515875, -39.1702728, 39.1972580
41: -32.0428085, 7.3592153, -32.0234756, 7.3438282, -38.0832367, 38.0704575
42: -30.1349678, -0.2491069, -30.1229324, -0.2690353, -23.1733017, 23.1797905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7849336, upper bound: 13.8059853
time: 30.80 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8059852, upper bound: 13.8059853
time: 34.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 67.23 seconds
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7105666, upper bound: 13.8054208
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7316378, upper bound: 13.8054208
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7270532, upper bound: 13.8054208
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7481174, upper bound: 13.8054208
IS_A1_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7697505, upper bound: 13.7849337
IS_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7697505, upper bound: 13.8059852
IS_A2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7849336, upper bound: 13.7701640
IS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7701640
IS_A2_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7849336, upper bound: 13.7866375
IS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7866375
IS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7468073, upper bound: 13.8054208
IS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7678765, upper bound: 13.8054208
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7632926, upper bound: 13.8054208
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7843539, upper bound: 13.8054208
IS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7684516
IS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.8059852, upper bound: 13.7895105
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.7849336, upper bound: 13.8059853
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.23
Output dim: 1, lower bound: -13.8059852, upper bound: 13.8059853

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.9554491, 25.8912201, -8.0176258, 25.8905334, -30.7449226, 30.7794228
1: -0.3004293, 26.7506199, -0.3260698, 26.7360382, -21.2993660, 21.3135605
2: -0.3514018, 25.7910385, -0.4033976, 25.7861385, -20.8193741, 20.8188400
3: -4.6982327, 22.4780750, -4.7475114, 22.4544773, -19.8874397, 19.8539658
4: -7.5936041, 22.4919167, -7.6388650, 22.4684753, -25.5451736, 25.5394821
5: -4.7803450, 24.8036842, -4.8361368, 24.7915478, -23.7219162, 23.6994934
6: -39.4118843, -4.2083349, -39.4179916, -4.1790981, -29.7314606, 29.7213364
7: -9.1197882, 23.3682899, -9.1598082, 23.3665314, -25.6766891, 25.6877289
8: -13.8290663, 19.9622784, -13.8590794, 19.9542160, -29.2212219, 29.2067146
9: -8.5249853, 22.4997540, -8.5590401, 22.5111179, -27.7800217, 27.7885437
10: -28.9456348, 17.8709679, -28.9654579, 17.9115906, -39.6595764, 39.6145554
11: -26.1758499, 6.8336949, -26.1685467, 6.8725076, -29.5566177, 29.6249084
12: -46.0602760, -8.3882828, -46.0571327, -8.2991295, -32.3418198, 32.2931213
13: -32.5221138, 13.2213173, -32.5597191, 13.2258453, -38.5247803, 38.5185242
14: -59.3982124, -2.1383181, -59.4150581, -2.0446253, -56.8126526, 56.8154907
15: -14.2100487, 18.7855949, -14.2218981, 18.7693539, -27.9776230, 27.9647675
16: -15.5812492, 22.2357502, -15.6223183, 22.2679729, -29.5585938, 29.5250778
17: -59.1299896, -6.9361782, -59.1184311, -6.9042873, -51.2694855, 51.2327728
18: -22.0083256, 16.2921448, -22.0040283, 16.3440990, -33.0403824, 33.0317841
19: -22.1083241, 6.0225201, -22.0921516, 6.0548635, -21.8940582, 21.9015198
20: -27.7847099, 0.5994291, -27.7730446, 0.6434102, -22.2643814, 22.3009415
21: -26.2651711, 7.3824091, -26.2572899, 7.4364467, -28.7967377, 28.8254890
22: -29.4095516, 5.2538328, -29.4065781, 5.2722726, -27.3892517, 27.4256897
23: -17.7768803, 12.2642155, -17.7581787, 12.2931023, -23.9228516, 23.9303436
24: -16.4390221, 13.6144600, -16.4205875, 13.6331739, -25.7919693, 25.7429581
25: -23.7507057, 8.8594103, -23.7184944, 8.8685341, -27.8024750, 27.8272018
26: -39.2595634, 3.9992828, -39.2615509, 4.0982766, -33.1254349, 33.1295929
27: -19.4617958, 14.8491220, -19.4745846, 14.8922844, -34.3540802, 34.3237076
28: -21.3195305, 11.4253197, -21.3079796, 11.4658604, -26.1875305, 26.1947365
29: -24.3428936, 6.3693380, -24.3385811, 6.3949766, -28.5315552, 28.5374146
30: -29.9701672, 5.8467064, -29.9481907, 5.8768797, -33.8944702, 33.8865814
31: -23.2988625, 7.6437855, -23.2850780, 7.6674271, -24.9571533, 24.9626579
32: -36.9750061, -2.6353397, -36.9762383, -2.6089935, -29.7750320, 29.7735291
33: -54.3874969, 0.0255737, -54.4458008, 0.0498199, -44.3825073, 44.4084473
34: -49.1068649, -7.0530028, -49.1360245, -7.0193491, -35.7936630, 35.7827148
35: -40.2577972, 3.8551226, -40.2965393, 3.8876057, -35.9483795, 35.9867325
36: -45.5453529, 1.2821016, -45.5711708, 1.3108664, -39.6246796, 39.6677551
37: -60.6493721, -9.9132414, -60.6513367, -9.8868656, -44.8197479, 44.8493958
38: -53.5808182, 3.9890795, -53.6202087, 4.0438709, -48.7086945, 48.7596588
39: -61.8692055, -4.6003151, -61.8815804, -4.5899057, -48.2256317, 48.2703629
40: -50.1546097, -9.1979160, -50.1534462, -9.1901951, -39.0705872, 39.0491943
41: -31.9759197, 7.2538304, -31.9953556, 7.2793078, -37.9419708, 37.9361801
42: -30.0833473, -0.3477468, -30.0873585, -0.3269963, -23.0649643, 23.0523262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6880222, upper bound: 13.8046101
time: 26.80 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7098236, upper bound: 13.8046801
time: 272.14 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.9610624, 25.9023895, -8.0557184, 25.9131031, -30.7648659, 30.8283157
1: -0.3029251, 26.7670612, -0.3581266, 26.7695847, -21.3120651, 21.3642235
2: -0.3545187, 25.8052483, -0.4284544, 25.8146381, -20.8326607, 20.8599892
3: -4.7002082, 22.4964657, -4.7744255, 22.4910336, -19.8962898, 19.9009438
4: -7.5975509, 22.5151711, -7.6764460, 22.5153275, -25.5745010, 25.6043701
5: -4.7854519, 24.8197365, -4.8667612, 24.8233051, -23.7419739, 23.7491112
6: -39.4142838, -4.2044125, -39.4284401, -4.1680689, -29.7420959, 29.7383881
7: -9.1254768, 23.3835011, -9.1944027, 23.3973465, -25.6911011, 25.7389946
8: -13.8307095, 19.9778023, -13.8880291, 19.9874134, -29.2283478, 29.2492256
9: -8.5276670, 22.5126305, -8.5776300, 22.5368557, -27.7971039, 27.8227768
10: -28.9481697, 17.8806419, -28.9837494, 17.9323120, -39.6700821, 39.6482468
11: -26.1887360, 6.8352442, -26.2008591, 6.8936276, -29.5928345, 29.6450500
12: -46.0663338, -8.3831348, -46.0710831, -8.2757912, -32.3604889, 32.3077698
13: -32.5340004, 13.2245274, -32.5856476, 13.2430668, -38.5385971, 38.5381775
14: -59.4083672, -2.1367111, -59.4404602, -2.0384474, -56.8273010, 56.8365021
15: -14.2130871, 18.8008366, -14.2488060, 18.8011627, -28.0031815, 28.0091438
16: -15.5850849, 22.2477245, -15.6433754, 22.2904587, -29.5624390, 29.5613937
17: -59.1498489, -6.9338942, -59.1685295, -6.8760147, -51.3191986, 51.2558441
18: -22.0133686, 16.2942963, -22.0257511, 16.3501282, -33.0475235, 33.0550766
19: -22.1299133, 6.0229454, -22.1374168, 6.0757413, -21.9358025, 21.9142647
20: -27.8040009, 0.6002126, -27.8125534, 0.6606622, -22.3052711, 22.3218384
21: -26.2827644, 7.3839192, -26.2954998, 7.4558501, -28.8316345, 28.8368378
22: -29.4255848, 5.2559958, -29.4411602, 5.2906799, -27.4204025, 27.4322319
23: -17.7974701, 12.2668285, -17.8009224, 12.3201733, -23.9747543, 23.9560928
24: -16.4570694, 13.6154270, -16.4575920, 13.6492786, -25.8294144, 25.7680664
25: -23.7859116, 8.8631639, -23.7893543, 8.9060574, -27.8799973, 27.8643951
26: -39.2740250, 4.0025167, -39.2924805, 4.1169205, -33.1613770, 33.1468582
27: -19.4658184, 14.8507261, -19.4836006, 14.8974590, -34.3632774, 34.3343277
28: -21.3423347, 11.4273396, -21.3544159, 11.4922190, -26.2407990, 26.2279396
29: -24.3535309, 6.3708630, -24.3634720, 6.4083052, -28.5501556, 28.5499229
30: -29.9859467, 5.8491797, -29.9814014, 5.9003100, -33.9355240, 33.9158554
31: -23.3196964, 7.6448145, -23.3296242, 7.6930728, -25.0067291, 24.9775887
32: -36.9777565, -2.6315651, -36.9859734, -2.5980406, -29.7880096, 29.7887192
33: -54.3999367, 0.0280256, -54.4713097, 0.0695066, -44.4127350, 44.4132080
34: -49.1086159, -7.0491080, -49.1453819, -7.0075712, -35.8084335, 35.7932358
35: -40.2655258, 3.8565435, -40.3127403, 3.8968153, -35.9712372, 35.9877319
36: -45.5607529, 1.2837439, -45.6021233, 1.3316212, -39.6489334, 39.6746750
37: -60.6699371, -9.9112139, -60.6942825, -9.8642712, -44.8523712, 44.8608170
38: -53.5940475, 3.9921150, -53.6487999, 4.0596609, -48.7301788, 48.7742691
39: -61.8894806, -4.5972815, -61.9225922, -4.5625992, -48.2638702, 48.2816315
40: -50.1577072, -9.1953430, -50.1678162, -9.1802158, -39.0799255, 39.0692291
41: -31.9786720, 7.2643538, -32.0094910, 7.3024197, -37.9648285, 37.9610367
42: -30.0924072, -0.3434329, -30.1062927, -0.3084149, -23.0917358, 23.0718536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7091130, upper bound: 13.8046101
time: 36.00 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7308952, upper bound: 13.8046801
time: 32.25 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.9607859, 25.9035435, -8.0421925, 25.9135246, -30.7490578, 30.8205109
1: -0.3015633, 26.7678299, -0.3428888, 26.7688332, -21.3016129, 21.3540154
2: -0.3534100, 25.8044128, -0.4180889, 25.8121796, -20.8178825, 20.8555984
3: -4.6993408, 22.4966908, -4.7625527, 22.4900227, -19.8811073, 19.9006958
4: -7.5951586, 22.5065689, -7.6533146, 22.4976845, -25.5558014, 25.5747528
5: -4.7824187, 24.8219090, -4.8529758, 24.8264027, -23.7193451, 23.7441788
6: -39.4132729, -4.2057056, -39.4197426, -4.1712618, -29.7412262, 29.7271805
7: -9.1219702, 23.3853207, -9.1766281, 23.3988495, -25.6818008, 25.7259979
8: -13.8301811, 19.9769287, -13.8750525, 19.9821930, -29.2218170, 29.2419052
9: -8.5271873, 22.5068417, -8.5693550, 22.5239258, -27.7859421, 27.8086700
10: -28.9525833, 17.8749046, -28.9801254, 17.9250488, -39.6766891, 39.6229706
11: -26.1884308, 6.8348370, -26.1929626, 6.8817215, -29.5792160, 29.6382408
12: -46.0718002, -8.3852444, -46.0797272, -8.2841921, -32.3707047, 32.2962341
13: -32.5242233, 13.2263947, -32.5661469, 13.2337580, -38.5313568, 38.5306473
14: -59.4226646, -2.1375351, -59.4652252, -2.0330124, -56.8539276, 56.8376923
15: -14.2115469, 18.7948170, -14.2303677, 18.7871513, -27.9836884, 27.9839363
16: -15.5841370, 22.2420425, -15.6305771, 22.2800083, -29.5660172, 29.5418167
17: -59.1521454, -6.9328976, -59.1622505, -6.8859320, -51.3246460, 51.2375031
18: -22.0223503, 16.2932739, -22.0304222, 16.3543625, -33.0638962, 33.0425797
19: -22.1253872, 6.0234737, -22.1244164, 6.0668716, -21.9306297, 21.9010506
20: -27.8021431, 0.6000710, -27.8064461, 0.6543689, -22.2997665, 22.3020630
21: -26.2781639, 7.3830905, -26.2813644, 7.4446063, -28.8209915, 28.8307266
22: -29.4317436, 5.2549176, -29.4499302, 5.2872696, -27.4287643, 27.4337463
23: -17.7954578, 12.2651434, -17.7944126, 12.3067379, -23.9582367, 23.9415169
24: -16.4593353, 13.6152782, -16.4580650, 13.6475124, -25.8291931, 25.7571869
25: -23.7741928, 8.8606634, -23.7626572, 8.8866158, -27.8513336, 27.8281860
26: -39.2831268, 4.0008469, -39.3065262, 4.1161447, -33.1713943, 33.1396942
27: -19.4741592, 14.8498440, -19.4977493, 14.8998947, -34.3740540, 34.3475952
28: -21.3396320, 11.4259453, -21.3460083, 11.4814110, -26.2284546, 26.2000313
29: -24.3620911, 6.3701315, -24.3752174, 6.4046507, -28.5622482, 28.5577774
30: -29.9889603, 5.8478584, -29.9830399, 5.8914037, -33.9346466, 33.8968201
31: -23.3176517, 7.6450057, -23.3200207, 7.6824689, -24.9967651, 24.9635277
32: -36.9820747, -2.6338286, -36.9901924, -2.6018066, -29.7881622, 29.7853699
33: -54.3921356, 0.0275822, -54.4554787, 0.0569906, -44.3862457, 44.4296570
34: -49.1178436, -7.0515399, -49.1578445, -7.0098243, -35.8175278, 35.7999878
35: -40.2600708, 3.8559189, -40.3017044, 3.8904285, -35.9569702, 35.9933014
36: -45.5556488, 1.2830210, -45.5909729, 1.3170795, -39.6444702, 39.6859512
37: -60.6658287, -9.9119396, -60.6834755, -9.8769302, -44.8296356, 44.8721848
38: -53.5912170, 3.9914846, -53.6406212, 4.0507126, -48.7331543, 48.7758560
39: -61.8835144, -4.5984583, -61.9094429, -4.5812473, -48.2294769, 48.2949905
40: -50.1646767, -9.1965437, -50.1724777, -9.1832314, -39.0756226, 39.0681763
41: -31.9773178, 7.2561035, -31.9983444, 7.2850552, -37.9496155, 37.9422379
42: -30.0878696, -0.3460245, -30.0962315, -0.3209352, -23.0757065, 23.0568848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7045256, upper bound: 13.8046101
time: 35.14 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7263107, upper bound: 13.8046801
time: 33.73 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.9664211, 25.9146881, -8.0802517, 25.9360962, -30.7690201, 30.8693619
1: -0.3040857, 26.7842865, -0.3749304, 26.8024101, -21.3143005, 21.4046936
2: -0.3565226, 25.8186245, -0.4431305, 25.8406849, -20.8311462, 20.8967323
3: -4.7012892, 22.5150967, -4.7895083, 22.5266056, -19.8899536, 19.9477043
4: -7.5991087, 22.5297794, -7.6908693, 22.5445461, -25.5851517, 25.6396484
5: -4.7875681, 24.8379936, -4.8835835, 24.8582020, -23.7393799, 23.7938309
6: -39.4156456, -4.2017651, -39.4302254, -4.1602135, -29.7518768, 29.7442169
7: -9.1276751, 23.4005165, -9.2112379, 23.4296684, -25.6962280, 25.7773132
8: -13.8318481, 19.9924831, -13.9040117, 20.0153923, -29.2289276, 29.2844505
9: -8.5298424, 22.5197506, -8.5879459, 22.5496387, -27.8030396, 27.8428764
10: -28.9551048, 17.8845844, -28.9983368, 17.9457684, -39.6871490, 39.6566925
11: -26.2013798, 6.8363748, -26.2252617, 6.9028478, -29.6153603, 29.6583786
12: -46.0778580, -8.3801231, -46.0936852, -8.2608776, -32.3893433, 32.3109512
13: -32.5360985, 13.2296038, -32.5920525, 13.2510328, -38.5452728, 38.5503082
14: -59.4327736, -2.1359730, -59.4906235, -2.0268593, -56.8685608, 56.8587494
15: -14.2146330, 18.8100319, -14.2573071, 18.8189850, -28.0092850, 28.0282898
16: -15.5879593, 22.2540913, -15.6516781, 22.3025169, -29.5698242, 29.5781555
17: -59.1720505, -6.9305840, -59.2124176, -6.8576527, -51.3743134, 51.2605286
18: -22.0273476, 16.2954445, -22.0521622, 16.3603535, -33.0710220, 33.0658646
19: -22.1470261, 6.0239229, -22.1697083, 6.0877280, -21.9723816, 21.9137840
20: -27.8214569, 0.6008253, -27.8459167, 0.6716585, -22.3406868, 22.3229637
21: -26.2957764, 7.3846235, -26.3196278, 7.4640303, -28.8559036, 28.8420830
22: -29.4478035, 5.2570362, -29.4845161, 5.3057408, -27.4599609, 27.4403076
23: -17.8160934, 12.2677336, -17.8371620, 12.3338137, -24.0101395, 23.9672890
24: -16.4773769, 13.6162481, -16.4950962, 13.6636171, -25.8666382, 25.7822723
25: -23.8093758, 8.8644657, -23.8335190, 8.9241152, -27.9287949, 27.8654022
26: -39.2975998, 4.0040522, -39.3374557, 4.1349010, -33.2073212, 33.1570053
27: -19.4781799, 14.8514681, -19.5068016, 14.9051342, -34.3833160, 34.3582687
28: -21.3624954, 11.4279642, -21.3924389, 11.5077744, -26.2817154, 26.2332726
29: -24.3727360, 6.3717041, -24.4001274, 6.4179373, -28.5808182, 28.5702362
30: -30.0048084, 5.8503022, -30.0162449, 5.9148393, -33.9756927, 33.9260559
31: -23.3384628, 7.6460295, -23.3645935, 7.7081032, -25.0463867, 24.9784431
32: -36.9848480, -2.6300201, -36.9999695, -2.5908251, -29.8012085, 29.8005676
33: -54.4045486, 0.0299883, -54.4808884, 0.0766993, -44.4164581, 44.4344330
34: -49.1195488, -7.0476799, -49.1672363, -6.9979944, -35.8323746, 35.8105316
35: -40.2678223, 3.8573799, -40.3179092, 3.8996582, -35.9798431, 35.9943619
36: -45.5710793, 1.2847185, -45.6219177, 1.3378410, -39.6687164, 39.6929321
37: -60.6864014, -9.9098969, -60.7264557, -9.8543339, -44.8622742, 44.8836594
38: -53.6044083, 3.9944706, -53.6693039, 4.0664444, -48.7546387, 48.7905502
39: -61.9038277, -4.5953817, -61.9504166, -4.5539389, -48.2677612, 48.3062820
40: -50.1678123, -9.1939554, -50.1868629, -9.1732750, -39.0849609, 39.0882111
41: -31.9801083, 7.2666111, -32.0124893, 7.3081350, -37.9723816, 37.9670792
42: -30.0969505, -0.3417277, -30.1151657, -0.3023677, -23.1024780, 23.0764389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7256151, upper bound: 13.8046101
time: 32.21 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7473756, upper bound: 13.8046801
time: 32.22 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.0914440, 25.9367886, -8.1445932, 25.9392757, -30.9304733, 30.9223328
1: -0.3874807, 26.8015480, -0.4172134, 26.8048496, -21.4474258, 21.4046898
2: -0.4244595, 25.8388577, -0.4784136, 25.8429260, -20.9375381, 20.8984718
3: -4.7559400, 22.5279045, -4.8177428, 22.5300999, -20.0097809, 19.9168549
4: -7.6935511, 22.5423050, -7.7388086, 22.5483398, -25.7094879, 25.6560135
5: -4.8562212, 24.8588200, -4.9194288, 24.8623905, -23.8588409, 23.7875671
6: -39.4281044, -4.1321192, -39.4347229, -4.1240759, -29.8224564, 29.8169403
7: -9.2280769, 23.4292526, -9.2632637, 23.4328403, -25.8237534, 25.8058472
8: -13.9221878, 20.0133018, -13.9508762, 20.0200615, -29.3544693, 29.3053780
9: -8.6230593, 22.5483990, -8.6343594, 22.5535698, -27.9022446, 27.8966446
10: -29.0354958, 17.9146042, -29.0411072, 17.9528770, -39.7612228, 39.7158737
11: -26.2386265, 6.8553352, -26.2463722, 6.9119134, -29.6457062, 29.7239685
12: -46.0944672, -8.2972536, -46.0994530, -8.2194328, -32.4291458, 32.4216385
13: -32.5855713, 13.2503166, -32.6196594, 13.2611437, -38.6137161, 38.6237106
14: -59.5170975, -2.1058569, -59.5355225, -2.0227642, -56.9283142, 56.9953461
15: -14.2553577, 18.8258629, -14.2762165, 18.8313656, -28.0873260, 28.0464630
16: -15.6858025, 22.2921467, -15.7004433, 22.3061829, -29.6614761, 29.6366882
17: -59.2360687, -6.9079094, -59.2468491, -6.8484001, -51.3694763, 51.4597168
18: -22.0554008, 16.3319149, -22.0624199, 16.3793945, -33.1070480, 33.1215820
19: -22.1679077, 6.0577717, -22.1773682, 6.1054335, -21.9408112, 22.0230026
20: -27.8427105, 0.6456618, -27.8516674, 0.6943555, -22.3261032, 22.4255486
21: -26.3191032, 7.4180593, -26.3302383, 7.4811225, -28.8530045, 28.9290466
22: -29.4837170, 5.3127742, -29.4940338, 5.3350272, -27.4667587, 27.5360298
23: -17.8346291, 12.3085213, -17.8432426, 12.3542786, -24.0004730, 24.0521393
24: -16.4946365, 13.6501656, -16.5019855, 13.6808310, -25.8599548, 25.8559418
25: -23.8328552, 8.9166174, -23.8400555, 8.9514809, -27.8941422, 27.9857559
26: -39.3373528, 4.0806074, -39.3447800, 4.1739545, -33.2301102, 33.2734070
27: -19.5057220, 14.8911772, -19.5186806, 14.9257383, -34.4314613, 34.4098587
28: -21.3868580, 11.4827604, -21.3985939, 11.5363846, -26.2817612, 26.3353729
29: -24.4001942, 6.4151449, -24.4125938, 6.4407067, -28.6047897, 28.6357231
30: -30.0150356, 5.8719530, -30.0231972, 5.9251795, -33.9634247, 33.9977417
31: -23.3625145, 7.6888561, -23.3730526, 7.7298789, -25.0240784, 25.1022224
32: -36.9983788, -2.5780487, -37.0056534, -2.5636311, -29.8537521, 29.8607788
33: -54.4437485, 0.1048203, -54.4887733, 0.1149406, -44.4906158, 44.4860611
34: -49.1633835, -6.9575453, -49.1711884, -6.9522996, -35.9131470, 35.8935852
35: -40.3112907, 3.9418364, -40.3246155, 3.9429026, -36.0523682, 36.0666504
36: -45.6219559, 1.3819904, -45.6280174, 1.3873215, -39.7505493, 39.7999420
37: -60.7284698, -9.8391857, -60.7355728, -9.8176794, -44.9285431, 44.9397812
38: -53.6712799, 4.1247969, -53.6799164, 4.1319218, -48.8748322, 48.9236832
39: -61.9436951, -4.5351095, -61.9630089, -4.5232172, -48.3226166, 48.3850327
40: -50.1893005, -9.1635828, -50.1973877, -9.1559038, -39.1397552, 39.1220169
41: -32.0091019, 7.3294463, -32.0175972, 7.3405218, -38.0446930, 38.0318680
42: -30.1128998, -0.2824421, -30.1189289, -0.2721863, -23.1367760, 23.1451187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_B2_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7472402, upper bound: 13.8051575
time: 36.70 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7690087, upper bound: 13.8052436
time: 34.09 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.1282682, 25.9747086, -8.0574160, 25.9072647, -30.9062309, 30.9315567
1: -0.4143448, 26.8432636, -0.3713684, 26.7629375, -21.3887253, 21.4599113
2: -0.4588332, 25.9199486, -0.3972168, 25.7907410, -20.8783340, 20.9787979
3: -4.7959828, 22.6087742, -4.7306385, 22.4613724, -19.8875160, 20.0029907
4: -7.7219386, 22.6078472, -7.6605577, 22.4923363, -25.6417274, 25.7321510
5: -4.8925657, 24.9306793, -4.8178062, 24.7945786, -23.7650223, 23.8679199
6: -39.4495087, -4.1268835, -39.4178276, -4.1501160, -29.8237762, 29.8034821
7: -9.2577734, 23.4453087, -9.1888399, 23.4007111, -25.7622223, 25.7909584
8: -13.9258184, 20.0676556, -13.8776255, 19.9637756, -29.2729454, 29.3441277
9: -8.6281681, 22.5373478, -8.6111708, 22.5005016, -27.8848801, 27.8275146
10: -29.0876083, 17.9366570, -28.9877930, 17.8578720, -39.7509460, 39.7070770
11: -26.4391823, 6.9131250, -26.1570053, 6.8666244, -29.9229889, 29.6346664
12: -46.1901054, -8.2697639, -46.0139160, -8.3944473, -32.4279099, 32.3318176
13: -32.6047745, 13.3175535, -32.6034164, 13.2228699, -38.6050415, 38.6269302
14: -59.6639442, -2.0572309, -59.4086304, -2.1526279, -57.0455627, 56.8577118
15: -14.2575874, 18.8541489, -14.2229519, 18.8009262, -28.0608368, 28.0615234
16: -15.7488003, 22.2861195, -15.6602249, 22.2694645, -29.6851273, 29.6164932
17: -59.4216270, -6.8694000, -59.1402779, -6.9200888, -51.6060181, 51.3153076
18: -22.1999474, 16.3700562, -21.9856911, 16.3464146, -33.2022018, 33.0989456
19: -22.2840691, 6.0997305, -22.1046143, 6.0724707, -22.1410408, 21.9208946
20: -27.9279156, 0.6812859, -27.7821541, 0.6366687, -22.4795227, 22.3261070
21: -26.4846439, 7.4746094, -26.2458305, 7.4189153, -29.0707397, 28.8362122
22: -29.5334930, 5.3319759, -29.4503059, 5.3108263, -27.5546951, 27.4164009
23: -17.9333305, 12.3511486, -17.7734203, 12.3209286, -24.1605682, 23.9911537
24: -16.5348244, 13.6877089, -16.4647942, 13.6863422, -25.9121399, 25.8290100
25: -23.8736534, 8.9412556, -23.7897568, 8.9259758, -28.0172653, 27.9056091
26: -39.4952965, 4.1337423, -39.2376442, 4.0551057, -33.4184799, 33.1575775
27: -19.5741997, 14.9430990, -19.4790115, 14.9367876, -34.5109863, 34.4221115
28: -21.4873085, 11.5309029, -21.3288097, 11.5015392, -26.4463882, 26.2682724
29: -24.4848938, 6.4380608, -24.3570862, 6.4144382, -28.7079544, 28.5418549
30: -30.1366444, 5.9387789, -29.9665203, 5.8998489, -34.1263580, 33.9789886
31: -23.4666901, 7.7248402, -23.3001842, 7.7090816, -25.2196579, 25.0148010
32: -37.0036621, -2.5804319, -36.9795265, -2.6238132, -29.8158875, 29.8292007
33: -54.4759254, 0.2268257, -54.4286804, 0.0554008, -44.4468536, 44.6138077
34: -49.1490479, -6.9323893, -49.1238441, -6.9865851, -35.9163513, 35.8710709
35: -40.3102417, 3.9742146, -40.2923851, 3.9241629, -36.0537567, 36.0821838
36: -45.6239624, 1.3951893, -45.6047745, 1.3566771, -39.7588043, 39.7407379
37: -60.7433586, -9.8348570, -60.6864853, -9.8622847, -44.8979797, 44.9251328
38: -53.6846886, 4.1395483, -53.6402435, 4.0888357, -48.8170776, 48.9556122
39: -61.9458122, -4.4541473, -61.9386368, -4.5621624, -48.3027344, 48.4387054
40: -50.1997719, -9.1309929, -50.1581535, -9.1765566, -39.0874481, 39.1606522
41: -32.0305939, 7.3435869, -31.9948902, 7.3179746, -38.0378571, 38.0249939
42: -30.1206627, -0.2691851, -30.1028042, -0.3043380, -23.1409302, 23.1288414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7693271
time: 28.79 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7694222
time: 31.91 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1527472, 25.9976940, -8.0627441, 25.9195976, -30.9473495, 30.9357147
1: -0.4311914, 26.8760986, -0.3725290, 26.7801399, -21.4291725, 21.4621277
2: -0.4735613, 25.9459629, -0.3992186, 25.8041401, -20.9150696, 20.9773254
3: -4.8110480, 22.6443710, -4.7317767, 22.4799747, -19.9342461, 19.9966698
4: -7.7363949, 22.6370735, -7.6621103, 22.5069618, -25.6770134, 25.7427139
5: -4.9093828, 24.9655991, -4.8198991, 24.8128185, -23.8097343, 23.8653564
6: -39.4513092, -4.1190510, -39.4191780, -4.1474886, -29.8295898, 29.8132706
7: -9.2746048, 23.4776382, -9.1910639, 23.4177361, -25.8005066, 25.7960930
8: -13.9418135, 20.0956554, -13.8787451, 19.9784431, -29.3081512, 29.3446617
9: -8.6384420, 22.5501537, -8.6133089, 22.5075741, -27.9050217, 27.8334274
10: -29.1022072, 17.9501190, -28.9947662, 17.8618908, -39.7593460, 39.7241669
11: -26.4635754, 6.9223819, -26.1695938, 6.8677583, -29.9363022, 29.6572266
12: -46.2126770, -8.2548523, -46.0254135, -8.3913841, -32.4310303, 32.3607788
13: -32.6111412, 13.3255615, -32.6055031, 13.2278948, -38.6172028, 38.6335068
14: -59.7140541, -2.0455742, -59.4331207, -2.1518030, -57.0677795, 56.8990021
15: -14.2660894, 18.8720093, -14.2244558, 18.8101883, -28.0799789, 28.0676384
16: -15.7570648, 22.2981720, -15.6630287, 22.2757759, -29.7018890, 29.6238594
17: -59.4654388, -6.8510389, -59.1624718, -6.9167700, -51.6107178, 51.3704987
18: -22.2264309, 16.3802624, -21.9996662, 16.3475552, -33.2129898, 33.1224823
19: -22.3163013, 6.1117668, -22.1217155, 6.0734038, -22.1405716, 21.9574776
20: -27.9612713, 0.6922445, -27.7996330, 0.6372976, -22.4806480, 22.3615227
21: -26.5087929, 7.4827633, -26.2588215, 7.4196386, -29.0759811, 28.8605156
22: -29.5768471, 5.3470078, -29.4725380, 5.3118773, -27.5627823, 27.4559402
23: -17.9695587, 12.3648081, -17.7920437, 12.3218107, -24.1717300, 24.0265160
24: -16.5723495, 13.7020378, -16.4851017, 13.6871614, -25.9263535, 25.8662262
25: -23.9178352, 8.9593325, -23.8132534, 8.9272518, -28.0182190, 27.9544296
26: -39.5402069, 4.1516814, -39.2611580, 4.0567131, -33.4285965, 33.2035675
27: -19.5974236, 14.9508152, -19.4913731, 14.9375210, -34.5349426, 34.4421883
28: -21.5253410, 11.5464439, -21.3489361, 11.5021658, -26.4516983, 26.3091888
29: -24.5215397, 6.4477339, -24.3762741, 6.4152632, -28.7283478, 28.5725403
30: -30.1714725, 5.9533215, -29.9853687, 5.9009933, -34.1366119, 34.0191727
31: -23.5016422, 7.7398386, -23.3189545, 7.7103057, -25.2204895, 25.0543938
32: -37.0175858, -2.5731893, -36.9866219, -2.6222720, -29.8277283, 29.8423615
33: -54.4855423, 0.2340126, -54.4333076, 0.0573578, -44.4680786, 44.6174927
34: -49.1708946, -6.9228220, -49.1348267, -6.9851656, -35.9336090, 35.8949661
35: -40.3154144, 3.9771051, -40.2946587, 3.9249992, -36.0603104, 36.0907211
36: -45.6437721, 1.4013071, -45.6151237, 1.3575935, -39.7770233, 39.7605591
37: -60.7755356, -9.8248968, -60.7029037, -9.8610630, -44.9208374, 44.9350815
38: -53.7051010, 4.1464052, -53.6506805, 4.0912123, -48.8332367, 48.9800415
39: -61.9737167, -4.4453678, -61.9530182, -4.5601969, -48.3273315, 48.4425735
40: -50.2188568, -9.1240215, -50.1682510, -9.1751738, -39.1064301, 39.1656876
41: -32.0335808, 7.3493657, -31.9963474, 7.3202505, -38.0438995, 38.0326004
42: -30.1295357, -0.2631340, -30.1073055, -0.3026481, -23.1455078, 23.1395988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7858128
time: 27.27 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7858960
time: 29.97 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.0728168, 25.9525604, -8.0577126, 25.8913727, -30.8483276, 30.8853188
1: -0.3750267, 26.8255386, -0.3497000, 26.7371197, -21.3668442, 21.4131927
2: -0.4593821, 25.8996983, -0.4453773, 25.7882347, -20.9160576, 20.9692421
3: -4.8195944, 22.5963173, -4.7959757, 22.4550858, -19.9682617, 19.9753113
4: -7.6902943, 22.5884476, -7.6749554, 22.4710083, -25.6302567, 25.6730423
5: -4.9063272, 24.9135914, -4.8854647, 24.7929039, -23.8194656, 23.8488617
6: -39.4414444, -4.1876030, -39.4224854, -4.1786642, -29.7702179, 29.7466965
7: -9.2115011, 23.4184666, -9.1887169, 23.3677082, -25.7601242, 25.7633972
8: -13.8943405, 20.0478554, -13.8825035, 19.9578304, -29.2878494, 29.3181152
9: -8.5495005, 22.5215569, -8.5654287, 22.5140533, -27.8519211, 27.7982674
10: -29.0182343, 17.9579716, -28.9692268, 17.9368172, -39.7802734, 39.7060471
11: -26.4043770, 6.9464641, -26.1717854, 6.9196024, -29.8323593, 29.7059250
12: -46.1809692, -8.2311630, -46.0590363, -8.2375412, -32.5196228, 32.4202003
13: -32.5551300, 13.3118696, -32.5666733, 13.2335434, -38.5671997, 38.6041336
14: -59.6133995, -1.9816952, -59.4277267, -1.9792309, -57.0789490, 56.9332275
15: -14.2431183, 18.8336220, -14.2216616, 18.7710228, -28.0286560, 27.9981537
16: -15.6664162, 22.2489548, -15.6328259, 22.2613583, -29.6472397, 29.5484581
17: -59.3663940, -6.8252411, -59.1245575, -6.8590946, -51.5607452, 51.3147736
18: -22.1844120, 16.3740273, -22.0074596, 16.3785496, -33.1788406, 33.0791168
19: -22.2658978, 6.1107550, -22.0981941, 6.0917764, -22.0921326, 21.9715004
20: -27.9098797, 0.6930437, -27.7781563, 0.6817732, -22.4367714, 22.3753624
21: -26.4631882, 7.5023227, -26.2640171, 7.4863200, -29.0493011, 28.9269180
22: -29.5081406, 5.3059759, -29.4118443, 5.2904577, -27.5126648, 27.4802094
23: -17.9190426, 12.3530636, -17.7633896, 12.3281574, -24.1027985, 24.0015411
24: -16.5217686, 13.6748409, -16.4237995, 13.6580429, -25.8790512, 25.7889404
25: -23.8380852, 8.9260130, -23.7213936, 8.8948069, -27.9127808, 27.8781052
26: -39.4669418, 4.1656160, -39.2652893, 4.1696415, -33.4117432, 33.2602386
27: -19.5657787, 14.9170399, -19.4826107, 14.9214315, -34.4872093, 34.3996506
28: -21.4683743, 11.5259418, -21.3155251, 11.5070992, -26.3793335, 26.2769852
29: -24.4688072, 6.4226351, -24.3443336, 6.4148846, -28.6993179, 28.5941582
30: -30.1295395, 5.9582491, -29.9518318, 5.9188085, -34.0949860, 33.9806290
31: -23.4492455, 7.7197418, -23.2921181, 7.6990423, -25.1429901, 25.0198975
32: -36.9983978, -2.6006846, -36.9801140, -2.6018748, -29.8113861, 29.8121872
33: -54.4706192, 0.1661682, -54.4790421, 0.0583630, -44.4467773, 44.5958023
34: -49.1325188, -7.0127354, -49.1405716, -7.0154777, -35.8554688, 35.8117599
35: -40.2738495, 3.8921337, -40.2983322, 3.8889351, -35.9788666, 36.0336685
36: -45.5713234, 1.3143415, -45.5742874, 1.3159742, -39.6866302, 39.7049026
37: -60.7026978, -9.8751106, -60.6540680, -9.8781910, -44.8565369, 44.9161987
38: -53.6186218, 4.0275850, -53.6235237, 4.0472689, -48.7240906, 48.8532944
39: -61.9063377, -4.4960308, -61.8942032, -4.5799007, -48.2672577, 48.3934631
40: -50.1934357, -9.1545734, -50.1576691, -9.1868343, -39.0990601, 39.1183090
41: -32.0093536, 7.2827487, -32.0011597, 7.2822804, -37.9799042, 37.9708176
42: -30.1048908, -0.3146353, -30.0910587, -0.3239493, -23.0975456, 23.0864677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7242655, upper bound: 13.8046101
time: 32.32 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7460642, upper bound: 13.8046801
time: 30.53 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.0784645, 25.9636917, -8.0957584, 25.9139652, -30.8683205, 30.9342117
1: -0.3775730, 26.8419685, -0.3817720, 26.7706833, -21.3795700, 21.4638634
2: -0.4625149, 25.9139404, -0.4704437, 25.8167572, -20.9293365, 21.0103836
3: -4.8215709, 22.6147270, -4.8229079, 22.4916401, -19.9771118, 20.0223160
4: -7.6942282, 22.6116753, -7.7125778, 22.5178490, -25.6595993, 25.7379532
5: -4.9114885, 24.9296265, -4.9160495, 24.8246937, -23.8395348, 23.8984871
6: -39.4437714, -4.1836529, -39.4329681, -4.1675835, -29.7808533, 29.7637100
7: -9.2172527, 23.4336624, -9.2233124, 23.3985615, -25.7745667, 25.8146896
8: -13.8959808, 20.0633621, -13.9114571, 19.9910088, -29.2949448, 29.3606453
9: -8.5521526, 22.5345039, -8.5840197, 22.5397873, -27.8689651, 27.8324661
10: -29.0207577, 17.9676704, -28.9874229, 17.9576130, -39.7907791, 39.7397614
11: -26.4173164, 6.9480162, -26.2041206, 6.9407244, -29.8685074, 29.7261124
12: -46.1870613, -8.2260132, -46.0729904, -8.2142677, -32.5382614, 32.4349365
13: -32.5669937, 13.3150959, -32.5925636, 13.2507248, -38.5810928, 38.6237564
14: -59.6235504, -1.9800482, -59.4531517, -1.9730415, -57.0936432, 56.9543152
15: -14.2461309, 18.8488007, -14.2485542, 18.8028526, -28.0542679, 28.0425034
16: -15.6702366, 22.2609615, -15.6539497, 22.2839432, -29.6510544, 29.5847931
17: -59.3862343, -6.8229055, -59.1746674, -6.8307705, -51.6103668, 51.3378448
18: -22.1894035, 16.3761921, -22.0292091, 16.3845768, -33.1860733, 33.1024170
19: -22.2874947, 6.1111689, -22.1434994, 6.1126723, -22.1338768, 21.9842644
20: -27.9291916, 0.6938357, -27.8176823, 0.6989951, -22.4776840, 22.3962250
21: -26.4807625, 7.5038223, -26.3021984, 7.5057173, -29.0842056, 28.9382477
22: -29.5242081, 5.3082066, -29.4464626, 5.3088698, -27.5438538, 27.4868011
23: -17.9396515, 12.3556747, -17.8061028, 12.3552284, -24.1547089, 24.0272751
24: -16.5397987, 13.6758385, -16.4607925, 13.6741238, -25.9165039, 25.8140335
25: -23.8732700, 8.9297886, -23.7922630, 8.9323320, -27.9902420, 27.9153252
26: -39.4814072, 4.1688499, -39.2962227, 4.1883044, -33.4476624, 33.2775345
27: -19.5697937, 14.9186268, -19.4916458, 14.9266186, -34.4964142, 34.4102707
28: -21.4911652, 11.5279541, -21.3619690, 11.5335083, -26.4325943, 26.3101959
29: -24.4794121, 6.4241962, -24.3692703, 6.4281564, -28.7178955, 28.6066818
30: -30.1453495, 5.9606996, -29.9849987, 5.9422712, -34.1360245, 34.0099182
31: -23.4700279, 7.7207341, -23.3366585, 7.7246695, -25.1926117, 25.0348473
32: -37.0011292, -2.5968971, -36.9898376, -2.5909171, -29.8243790, 29.8273697
33: -54.4829521, 0.1685772, -54.5045357, 0.0780735, -44.4769897, 44.6005325
34: -49.1343155, -7.0087981, -49.1499596, -7.0035934, -35.8702545, 35.8223343
35: -40.2816277, 3.8935623, -40.3145485, 3.8981867, -36.0016708, 36.0346603
36: -45.5867348, 1.3159914, -45.6052170, 1.3367767, -39.7108688, 39.7118988
37: -60.7232208, -9.8730659, -60.6969681, -9.8555450, -44.8891907, 44.9276733
38: -53.6317825, 4.0306044, -53.6522369, 4.0629635, -48.7455444, 48.8679504
39: -61.9266663, -4.4929428, -61.9351616, -4.5525217, -48.3054504, 48.4047623
40: -50.1965332, -9.1519861, -50.1720467, -9.1768675, -39.1083832, 39.1383667
41: -32.0121498, 7.2932591, -32.0152512, 7.3054113, -38.0027313, 37.9957352
42: -30.1139469, -0.3103065, -30.1099930, -0.3053422, -23.1243362, 23.1060333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7453554, upper bound: 13.8046101
time: 26.56 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7671341, upper bound: 13.8046801
time: 28.26 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.0781536, 25.9648724, -8.0822306, 25.9143620, -30.8524933, 30.9264526
1: -0.3761835, 26.8427315, -0.3665257, 26.7699604, -21.3691254, 21.4536514
2: -0.4613500, 25.9131145, -0.4600692, 25.8143024, -20.9146080, 21.0060158
3: -4.8206902, 22.6149464, -4.8110833, 22.4906311, -19.9619560, 20.0220871
4: -7.6918058, 22.6030769, -7.6894250, 22.5002327, -25.6408844, 25.7083511
5: -4.9084616, 24.9318256, -4.9023232, 24.8278217, -23.8168869, 23.8935738
6: -39.4427834, -4.1849594, -39.4242668, -4.1708221, -29.7799835, 29.7524796
7: -9.2137794, 23.4354954, -9.2055435, 23.4000568, -25.7652588, 25.8017044
8: -13.8954535, 20.0625038, -13.8984995, 19.9857826, -29.2884293, 29.3533173
9: -8.5516586, 22.5287056, -8.5757542, 22.5268764, -27.8578110, 27.8183975
10: -29.0252075, 17.9619274, -28.9838181, 17.9503212, -39.7974167, 39.7144241
11: -26.4170094, 6.9476466, -26.1961803, 6.9288182, -29.8549347, 29.7192345
12: -46.1925430, -8.2280655, -46.0816803, -8.2225895, -32.5485077, 32.4233894
13: -32.5572090, 13.3168926, -32.5729828, 13.2415161, -38.5738525, 38.6162491
14: -59.6378250, -1.9809589, -59.4777832, -1.9675369, -57.1203003, 56.9554901
15: -14.2446022, 18.8428555, -14.2301350, 18.7888393, -28.0347900, 28.0173416
16: -15.6692467, 22.2553082, -15.6411552, 22.2734089, -29.6546631, 29.5652084
17: -59.3886185, -6.8219891, -59.1683998, -6.8407326, -51.6159668, 51.3193817
18: -22.1984138, 16.3751373, -22.0339088, 16.3888283, -33.2024002, 33.0898819
19: -22.2829857, 6.1117139, -22.1304817, 6.1037874, -22.1287193, 21.9710312
20: -27.9273262, 0.6936526, -27.8115196, 0.6927161, -22.4721985, 22.3764687
21: -26.4761696, 7.5030336, -26.2880859, 7.4944367, -29.0735931, 28.9321480
22: -29.5303421, 5.3070803, -29.4551964, 5.3055353, -27.5522308, 27.4882812
23: -17.9376297, 12.3539696, -17.7996254, 12.3417549, -24.1381912, 24.0126877
24: -16.5420742, 13.6756783, -16.4612789, 13.6723747, -25.9163208, 25.8031235
25: -23.8615494, 8.9272814, -23.7655468, 8.9128923, -27.9616318, 27.8791008
26: -39.4905243, 4.1672087, -39.3102722, 4.1875687, -33.4577026, 33.2703323
27: -19.5781651, 14.9177284, -19.5057812, 14.9291191, -34.5072861, 34.4235077
28: -21.4884872, 11.5265379, -21.3535538, 11.5226364, -26.4203033, 26.2823067
29: -24.4879761, 6.4234262, -24.3809605, 6.4245081, -28.7300186, 28.6145096
30: -30.1483459, 5.9593964, -29.9866428, 5.9333668, -34.1352081, 33.9908676
31: -23.4680138, 7.7209816, -23.3270569, 7.7140398, -25.1826172, 25.0207825
32: -37.0054550, -2.5991640, -36.9940300, -2.5946708, -29.8245773, 29.8240204
33: -54.4752159, 0.1681137, -54.4886322, 0.0655632, -44.4505157, 44.6169968
34: -49.1435623, -7.0112915, -49.1624374, -7.0058937, -35.8793564, 35.8290253
35: -40.2761307, 3.8929710, -40.3035202, 3.8918028, -35.9874039, 36.0402145
36: -45.5816193, 1.3152723, -45.5940704, 1.3222189, -39.7064438, 39.7231216
37: -60.7191162, -9.8738194, -60.6861725, -9.8682365, -44.8665009, 44.9390640
38: -53.6289864, 4.0299044, -53.6439667, 4.0540943, -48.7485352, 48.8695145
39: -61.9206085, -4.4940310, -61.9220772, -4.5711727, -48.2711029, 48.4181290
40: -50.2034988, -9.1532059, -50.1767654, -9.1799183, -39.1040344, 39.1372757
41: -32.0108070, 7.2849474, -32.0041542, 7.2879901, -37.9875717, 37.9768982
42: -30.1093750, -0.3129683, -30.0999298, -0.3178802, -23.1082993, 23.0910416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7407686, upper bound: 13.8046101
time: 46.45 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7625503, upper bound: 13.8046801
time: 31.11 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.0838041, 25.9760189, -8.1202574, 25.9369698, -30.8724442, 30.9753113
1: -0.3786979, 26.8591843, -0.3985853, 26.8035164, -21.3817978, 21.5043259
2: -0.4644961, 25.9273262, -0.4851189, 25.8427944, -20.9278717, 21.0471153
3: -4.8226485, 22.6333656, -4.8379827, 22.5272236, -19.9707947, 20.0690842
4: -7.6957822, 22.6263199, -7.7270222, 22.5470562, -25.6702118, 25.7732315
5: -4.9135408, 24.9478798, -4.9328580, 24.8595924, -23.8369827, 23.9431953
6: -39.4451370, -4.1810074, -39.4347229, -4.1597185, -29.7906418, 29.7695465
7: -9.2194653, 23.4507217, -9.2401533, 23.4308872, -25.7796898, 25.8529930
8: -13.8971138, 20.0779839, -13.9274273, 20.0189934, -29.2955246, 29.3958511
9: -8.5543289, 22.5415535, -8.5943365, 22.5526237, -27.8748703, 27.8525620
10: -29.0277309, 17.9716110, -29.0020218, 17.9710369, -39.8078384, 39.7482147
11: -26.4299335, 6.9491453, -26.2285042, 6.9499693, -29.8910980, 29.7394257
12: -46.1985931, -8.2229195, -46.0956268, -8.1993675, -32.5671005, 32.4381294
13: -32.5690880, 13.3201342, -32.5989113, 13.2587023, -38.5877075, 38.6359024
14: -59.6479759, -1.9793415, -59.5032501, -1.9613647, -57.1348572, 56.9765320
15: -14.2476521, 18.8580246, -14.2570114, 18.8206825, -28.0603714, 28.0616684
16: -15.6730385, 22.2672539, -15.6622276, 22.2959518, -29.6584549, 29.6015244
17: -59.4083786, -6.8196115, -59.2185593, -6.8124895, -51.6656036, 51.3425140
18: -22.2033844, 16.3772888, -22.0556469, 16.3948288, -33.2095718, 33.1132126
19: -22.3045368, 6.1121502, -22.1757431, 6.1246929, -22.1704712, 21.9837837
20: -27.9466457, 0.6944156, -27.8510437, 0.7099843, -22.5130692, 22.3973656
21: -26.4938145, 7.5045333, -26.3262768, 7.5138574, -29.1084976, 28.9434967
22: -29.5464249, 5.3092175, -29.4897785, 5.3239036, -27.5833740, 27.4948349
23: -17.9582596, 12.3565483, -17.8423328, 12.3688774, -24.1901016, 24.0384560
24: -16.5601215, 13.6766758, -16.4982986, 13.6884499, -25.9537277, 25.8283005
25: -23.8967247, 8.9310694, -23.8364601, 8.9503899, -28.0390778, 27.9163475
26: -39.5050201, 4.1704855, -39.3411942, 4.2063260, -33.4935837, 33.2876968
27: -19.5821819, 14.9193478, -19.5148373, 14.9343090, -34.5164909, 34.4341850
28: -21.5113087, 11.5285473, -21.3999939, 11.5490637, -26.4735641, 26.3155670
29: -24.4986420, 6.4249735, -24.4058914, 6.4378052, -28.7485352, 28.6270218
30: -30.1641655, 5.9618425, -30.0198383, 5.9568610, -34.1762238, 34.0201187
31: -23.4888268, 7.7219887, -23.3715973, 7.7397037, -25.2322083, 25.0357056
32: -37.0082359, -2.5953717, -37.0038071, -2.5836916, -29.8375702, 29.8391876
33: -54.4876480, 0.1705818, -54.5141640, 0.0852766, -44.4807129, 44.6217957
34: -49.1452522, -7.0073853, -49.1718102, -6.9940314, -35.8941574, 35.8396072
35: -40.2838478, 3.8944435, -40.3197441, 3.9010277, -36.0102768, 36.0412216
36: -45.5970840, 1.3169651, -45.6249847, 1.3429546, -39.7306976, 39.7300644
37: -60.7396469, -9.8718090, -60.7291527, -9.8456507, -44.8991089, 44.9504623
38: -53.6422119, 4.0329866, -53.6726494, 4.0697708, -48.7700043, 48.8841782
39: -61.9409904, -4.4910250, -61.9630699, -4.5437584, -48.3092804, 48.4293976
40: -50.2066269, -9.1506195, -50.1911011, -9.1699009, -39.1134338, 39.1572876
41: -32.0135765, 7.2955346, -32.0182571, 7.3110976, -38.0104065, 38.0017471
42: -30.1184692, -0.3086815, -30.1188660, -0.2992835, -23.1350632, 23.1105804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7618571, upper bound: 13.8046101
time: 26.16 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7836123, upper bound: 13.8046801
time: 33.15 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.1462021, 25.9525509, -8.1736517, 25.9166546, -30.9439621, 31.0042038
1: -0.4131594, 26.8100929, -0.4371872, 26.7722931, -21.4237823, 21.4893723
2: -0.4926696, 25.8930302, -0.5153146, 25.8174210, -20.9564056, 21.0370560
3: -4.8352809, 22.5740566, -4.8631926, 22.4937019, -19.9968719, 20.0357170
4: -7.7381234, 22.5627651, -7.7694521, 22.5130005, -25.6943588, 25.7496643
5: -4.9347272, 24.9020786, -4.9615421, 24.8295059, -23.8620796, 23.9194756
6: -39.4454041, -4.1302500, -39.4355049, -4.1302872, -29.8382950, 29.8218536
7: -9.2684650, 23.4162827, -9.2842827, 23.4018383, -25.8176117, 25.8619919
8: -13.9424801, 20.0377140, -13.9715509, 19.9934616, -29.3433342, 29.4091415
9: -8.6185856, 22.5317001, -8.6358900, 22.5365067, -27.9196777, 27.8833618
10: -29.0751991, 17.9674110, -29.0353851, 17.9645100, -39.8396988, 39.7798920
11: -26.4105358, 6.9377360, -26.2240982, 6.9563079, -29.8880539, 29.7461929
12: -46.1786690, -8.1783581, -46.0838089, -8.1660500, -32.5889740, 32.5012054
13: -32.5862923, 13.3156395, -32.6126328, 13.2605495, -38.6243591, 38.6888733
14: -59.6568031, -1.9670391, -59.5135880, -1.9596806, -57.1514130, 57.0573120
15: -14.2530537, 18.8242111, -14.2714281, 18.8085976, -28.0748520, 28.0481834
16: -15.7415581, 22.2708340, -15.7043476, 22.2813644, -29.6969604, 29.6488571
17: -59.3786545, -6.8435602, -59.2108803, -6.8087568, -51.6331177, 51.4369507
18: -22.1832886, 16.3975925, -22.0468674, 16.4105930, -33.2114868, 33.1382294
19: -22.2479610, 6.1131258, -22.1446953, 6.1409764, -22.1266632, 22.0146599
20: -27.8950062, 0.7109985, -27.8199730, 0.7313576, -22.4765091, 22.4236298
21: -26.4547844, 7.5103502, -26.3063717, 7.5287848, -29.0890198, 28.9712753
22: -29.5044079, 5.3315516, -29.4610558, 5.3500357, -27.5755157, 27.5198364
23: -17.8978882, 12.3566370, -17.8092308, 12.3858356, -24.1435242, 24.0360336
24: -16.5028934, 13.6802225, -16.4668884, 13.7039309, -25.9077911, 25.8273163
25: -23.8051529, 8.9276304, -23.7843170, 8.9726934, -27.9661560, 27.9103966
26: -39.4689789, 4.2103271, -39.3104553, 4.2404518, -33.4889908, 33.3222427
27: -19.5774212, 14.9462013, -19.5103760, 14.9525681, -34.5299911, 34.4565773
28: -21.4512138, 11.5414190, -21.3632107, 11.5750341, -26.4350433, 26.3234711
29: -24.4645958, 6.4454885, -24.3884716, 6.4582367, -28.7396240, 28.6432266
30: -30.1063404, 5.9454432, -29.9921207, 5.9635644, -34.1245041, 34.0105438
31: -23.4333725, 7.7241631, -23.3404999, 7.7592239, -25.1941757, 25.0702858
32: -36.9980736, -2.5615773, -36.9996834, -2.5617757, -29.8631210, 29.8732986
33: -54.4917259, 0.2185907, -54.5049858, 0.1191254, -44.5289154, 44.6394043
34: -49.1578903, -6.9385967, -49.1630669, -6.9537725, -35.9472198, 35.8839035
35: -40.3059845, 3.9667969, -40.3163643, 3.9420099, -36.0752029, 36.0821915
36: -45.5972328, 1.3872223, -45.6053505, 1.3898602, -39.7872925, 39.7929840
37: -60.7066040, -9.8335905, -60.7012939, -9.8123178, -44.9311523, 44.9639740
38: -53.6599503, 4.1407328, -53.6596985, 4.1298656, -48.8594360, 48.9713669
39: -61.9119873, -4.4669714, -61.9410439, -4.5181332, -48.3282013, 48.4659195
40: -50.1946640, -9.1371183, -50.1883774, -9.1565895, -39.1292572, 39.1768112
41: -32.0254517, 7.3295131, -32.0191574, 7.3306932, -38.0517273, 38.0361023
42: -30.1066265, -0.2740211, -30.1090736, -0.2751112, -23.1452484, 23.1417122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7675997
time: 42.27 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7677092
time: 26.77 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1842995, 25.9751434, -8.1792946, 25.9277935, -30.9928818, 31.0241394
1: -0.4452820, 26.8436356, -0.4397221, 26.7887287, -21.4744606, 21.5020599
2: -0.5177388, 25.9215240, -0.5184078, 25.8316116, -20.9975052, 21.0503235
3: -4.8621922, 22.6106091, -4.8651748, 22.5120525, -20.0438576, 20.0445786
4: -7.7757354, 22.6096344, -7.7733612, 22.5362301, -25.7592659, 25.7790108
5: -4.9653616, 24.9338398, -4.9666657, 24.8455696, -23.9117432, 23.9395447
6: -39.4558792, -4.1192150, -39.4378777, -4.1263161, -29.8553467, 29.8324814
7: -9.3030500, 23.4470978, -9.2899790, 23.4170227, -25.8689232, 25.8764610
8: -13.9714575, 20.0708961, -13.9731741, 20.0089951, -29.3859024, 29.4162064
9: -8.6371593, 22.5574131, -8.6385384, 22.5494118, -27.9539032, 27.9004364
10: -29.0934448, 17.9881916, -29.0378990, 17.9741802, -39.8735428, 39.7903442
11: -26.4427948, 6.9588499, -26.2370491, 6.9578733, -29.9081116, 29.7824097
12: -46.1926193, -8.1549950, -46.0898933, -8.1609459, -32.6037598, 32.5198975
13: -32.6122093, 13.3328629, -32.6244507, 13.2637844, -38.6441040, 38.7027512
14: -59.6821327, -1.9608383, -59.5237579, -1.9580956, -57.1723785, 57.0719910
15: -14.2799854, 18.8560333, -14.2744493, 18.8237991, -28.1192322, 28.0737457
16: -15.7626286, 22.2933731, -15.7081757, 22.2933044, -29.7333145, 29.6527100
17: -59.4286308, -6.8152990, -59.2306976, -6.8064108, -51.6561508, 51.4865417
18: -22.2050495, 16.4035645, -22.0518341, 16.4127274, -33.2348404, 33.1453934
19: -22.2932110, 6.1339846, -22.1663399, 6.1413989, -22.1393623, 22.0564041
20: -27.9345303, 0.7282701, -27.8392963, 0.7321310, -22.4973679, 22.4645424
21: -26.4929543, 7.5297647, -26.3239784, 7.5302978, -29.1003494, 29.0062103
22: -29.5389977, 5.3499470, -29.4770889, 5.3522182, -27.5821304, 27.5510483
23: -17.9405594, 12.3837566, -17.8298416, 12.3884516, -24.1692505, 24.0879593
24: -16.5398846, 13.6962700, -16.4848976, 13.7048855, -25.9328690, 25.8647308
25: -23.8760071, 8.9651566, -23.8194981, 8.9764709, -28.0034027, 27.9878883
26: -39.4998322, 4.2290859, -39.3249741, 4.2437458, -33.5063019, 33.3581467
27: -19.5864792, 14.9513760, -19.5143509, 14.9541883, -34.5406685, 34.4657288
28: -21.4976463, 11.5677834, -21.3860054, 11.5770073, -26.4682312, 26.3767319
29: -24.4895077, 6.4587660, -24.3991318, 6.4597974, -28.7521210, 28.6617966
30: -30.1395226, 5.9688854, -30.0079575, 5.9660492, -34.1537094, 34.0515671
31: -23.4778709, 7.7497935, -23.3613091, 7.7602396, -25.2090607, 25.1198540
32: -37.0077972, -2.5505533, -37.0024567, -2.5579915, -29.8782654, 29.8863297
33: -54.5171928, 0.2382994, -54.5174065, 0.1214819, -44.5336456, 44.6696777
34: -49.1672516, -6.9267921, -49.1648064, -6.9498396, -35.9577637, 35.8987579
35: -40.3221779, 3.9760513, -40.3241730, 3.9434185, -36.0762329, 36.1050110
36: -45.6281738, 1.4080067, -45.6207924, 1.3914690, -39.7942734, 39.8172760
37: -60.7496262, -9.8109970, -60.7218361, -9.8102589, -44.9426422, 44.9966583
38: -53.6886444, 4.1564398, -53.6729050, 4.1328707, -48.8740845, 48.9929123
39: -61.9529991, -4.4396076, -61.9613838, -4.5150537, -48.3395844, 48.5042267
40: -50.2090683, -9.1271172, -50.1914902, -9.1539822, -39.1493225, 39.1861420
41: -32.0396118, 7.3525939, -32.0219345, 7.3412304, -38.0766602, 38.0588760
42: -30.1255512, -0.2554302, -30.1181469, -0.2708173, -23.1647797, 23.1685410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7886723
time: 30.71 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7887686
time: 32.72 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.2037830, 25.9874783, -8.1459188, 25.9170246, -31.0064507, 30.9869080
1: -0.4596968, 26.8607731, -0.4086390, 26.7716503, -21.4771919, 21.4786301
2: -0.5295897, 25.9337311, -0.4951143, 25.8161392, -21.0017776, 21.0269089
3: -4.8753676, 22.6279659, -4.8392859, 22.4939575, -20.0497704, 20.0232086
4: -7.7864351, 22.6167717, -7.7371292, 22.5028629, -25.7416687, 25.7482834
5: -4.9772325, 24.9532890, -4.9381070, 24.8314095, -23.9167595, 23.9069519
6: -39.4553986, -4.1143236, -39.4286690, -4.1356764, -29.8464432, 29.8293076
7: -9.3143005, 23.4648857, -9.2574387, 23.4025574, -25.8698235, 25.8532410
8: -13.9858732, 20.0842133, -13.9452705, 19.9895973, -29.3933601, 29.3948059
9: -8.6450005, 22.5585289, -8.6219568, 22.5296001, -27.9437332, 27.8853760
10: -29.1056519, 17.9939060, -29.0264473, 17.9554596, -39.8540039, 39.7911377
11: -26.4543991, 6.9666576, -26.2172279, 6.9377613, -29.8978653, 29.7722511
12: -46.2097092, -8.1451187, -46.0868988, -8.1812620, -32.5872879, 32.5350723
13: -32.6085663, 13.3377609, -32.5988312, 13.2514639, -38.6426849, 38.6893005
14: -59.7229538, -1.9507465, -59.5218773, -1.9636250, -57.1845551, 57.0874939
15: -14.2855682, 18.8592167, -14.2488909, 18.8006592, -28.1017303, 28.0465393
16: -15.7673302, 22.2954693, -15.6896877, 22.2750473, -29.7230988, 29.6469154
17: -59.4529114, -6.7988644, -59.2025452, -6.8318644, -51.6410065, 51.4887848
18: -22.2267532, 16.4121494, -22.0438461, 16.4073219, -33.2350464, 33.1490402
19: -22.3040352, 6.1457219, -22.1379490, 6.1213670, -22.1309509, 22.0464439
20: -27.9495354, 0.7385488, -27.8162708, 0.7154002, -22.4801559, 22.4564896
21: -26.4998989, 7.5365362, -26.2983398, 7.5115099, -29.0950546, 28.9947014
22: -29.5675163, 5.3629203, -29.4634361, 5.3347492, -27.5836792, 27.5592804
23: -17.9567490, 12.3949032, -17.8051720, 12.3621073, -24.1514740, 24.0746040
24: -16.5606766, 13.7096977, -16.4668694, 13.6896400, -25.9254837, 25.8610458
25: -23.8868275, 8.9796286, -23.7702923, 8.9400167, -27.9659805, 27.9604340
26: -39.5309448, 4.2437811, -39.3170052, 4.2266192, -33.4991074, 33.3682022
27: -19.6063690, 14.9579439, -19.5170097, 14.9492474, -34.5556183, 34.4749527
28: -21.5135174, 11.5814466, -21.3590622, 11.5511713, -26.4385376, 26.3661728
29: -24.5178890, 6.4669619, -24.3909645, 6.4472256, -28.7662354, 28.6675644
30: -30.1592522, 5.9811511, -29.9928627, 5.9435663, -34.1306381, 34.0547867
31: -23.4922066, 7.7639036, -23.3353901, 7.7357168, -25.1921997, 25.1126518
32: -37.0196152, -2.5464563, -36.9991760, -2.5681324, -29.8755569, 29.8858643
33: -54.5155563, 0.2431393, -54.4954376, 0.1036797, -44.5452576, 44.6479874
34: -49.1874390, -6.9208951, -49.1663628, -6.9604545, -35.9611893, 35.9110184
35: -40.3215561, 3.9775352, -40.3082848, 3.9349012, -36.0738907, 36.0986481
36: -45.6331673, 1.4126396, -45.5995407, 1.3715172, -39.8040314, 39.8143158
37: -60.7618179, -9.8030815, -60.6947060, -9.8316879, -44.9536743, 44.9742050
38: -53.6983566, 4.1604662, -53.6521873, 4.1193161, -48.8760376, 48.9954300
39: -61.9618645, -4.4338207, -61.9333267, -4.5406036, -48.3518066, 48.4708862
40: -50.2256126, -9.1212521, -50.1866226, -9.1640625, -39.1547852, 39.1752243
41: -32.0398903, 7.3483114, -32.0091438, 7.3198652, -38.0564575, 38.0450745
42: -30.1255951, -0.2535353, -30.1034431, -0.2878795, -23.1460037, 23.1562920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7623970, upper bound: 13.8051575
time: 35.06 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7841912, upper bound: 13.8052437
time: 33.79 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.2094421, 25.9986343, -8.1839962, 25.9396076, -31.0264130, 31.0357819
1: -0.4622526, 26.8772354, -0.4407210, 26.8051834, -21.4898987, 21.5293007
2: -0.5326719, 25.9479275, -0.5201268, 25.8446388, -21.0150757, 21.0680275
3: -4.8773365, 22.6463890, -4.8662500, 22.5305157, -20.0586548, 20.0702133
4: -7.7903709, 22.6399956, -7.7747154, 22.5497036, -25.7709923, 25.8131790
5: -4.9823518, 24.9693718, -4.9686241, 24.8632011, -23.9368401, 23.9565849
6: -39.4577713, -4.1103716, -39.4391136, -4.1246414, -29.8571243, 29.8463287
7: -9.3200312, 23.4800777, -9.2920341, 23.4333954, -25.8842850, 25.9044876
8: -13.9875221, 20.0997086, -13.9742317, 20.0228157, -29.4004936, 29.4373436
9: -8.6476774, 22.5714302, -8.6405325, 22.5553322, -27.9608002, 27.9195709
10: -29.1082382, 18.0035973, -29.0446701, 17.9762573, -39.8644485, 39.8248749
11: -26.4673061, 6.9682045, -26.2494946, 6.9588995, -29.9340210, 29.7923813
12: -46.2157860, -8.1399584, -46.1008453, -8.1579752, -32.6058578, 32.5498199
13: -32.6203728, 13.3409824, -32.6246986, 13.2686605, -38.6565781, 38.7089081
14: -59.7330704, -1.9491272, -59.5473595, -1.9574213, -57.1992035, 57.1085968
15: -14.2886162, 18.8744259, -14.2758131, 18.8324680, -28.1273575, 28.0909042
16: -15.7711468, 22.3074646, -15.7108212, 22.2975330, -29.7268982, 29.6832352
17: -59.4727783, -6.7965107, -59.2526398, -6.8036375, -51.6906433, 51.5118408
18: -22.2317276, 16.4142914, -22.0655918, 16.4133434, -33.2422409, 33.1723557
19: -22.3256664, 6.1461248, -22.1832504, 6.1422558, -22.1727104, 22.0591888
20: -27.9688396, 0.7393465, -27.8557568, 0.7326317, -22.5210876, 22.4773636
21: -26.5174809, 7.5379815, -26.3365955, 7.5309267, -29.1299438, 29.0060654
22: -29.5835857, 5.3650570, -29.4980621, 5.3531628, -27.6148300, 27.5658646
23: -17.9773273, 12.3975315, -17.8478985, 12.3892212, -24.2033615, 24.1003723
24: -16.5787125, 13.7106371, -16.5038948, 13.7056503, -25.9629288, 25.8861618
25: -23.9219990, 8.9834290, -23.8411751, 8.9775314, -28.0434647, 27.9976807
26: -39.5454102, 4.2470284, -39.3479309, 4.2452679, -33.5349731, 33.3855209
27: -19.6103783, 14.9595680, -19.5260410, 14.9544411, -34.5648193, 34.4856110
28: -21.5363216, 11.5834522, -21.4054756, 11.5775604, -26.4917755, 26.3994370
29: -24.5285606, 6.4685268, -24.4159164, 6.4605269, -28.7848358, 28.6800804
30: -30.1750622, 5.9836287, -30.0261116, 5.9669900, -34.1717224, 34.0840302
31: -23.5130005, 7.7649007, -23.3799629, 7.7613506, -25.2418137, 25.1275978
32: -37.0223465, -2.5427279, -37.0089645, -2.5571642, -29.8885345, 29.9010315
33: -54.5279922, 0.2455463, -54.5208893, 0.1233578, -44.5754852, 44.6527481
34: -49.1891899, -6.9169941, -49.1757011, -6.9486423, -35.9759827, 35.9216003
35: -40.3292732, 3.9789724, -40.3244743, 3.9442158, -36.0967178, 36.0996094
36: -45.6486015, 1.4143076, -45.6304817, 1.3923855, -39.8283005, 39.8212433
37: -60.7823448, -9.8009329, -60.7376595, -9.8090649, -44.9862976, 44.9856873
38: -53.7114944, 4.1635075, -53.6808052, 4.1350622, -48.8975067, 49.0100555
39: -61.9822159, -4.4308090, -61.9743767, -4.5131750, -48.3899994, 48.4821701
40: -50.2287521, -9.1187057, -50.2009735, -9.1541672, -39.1641388, 39.1952209
41: -32.0427017, 7.3588877, -32.0232582, 7.3429432, -38.0792542, 38.0699615
42: -30.1346989, -0.2492132, -30.1224022, -0.2693048, -23.1727676, 23.1758385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7834814, upper bound: 13.8051575
time: 37.67 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8052435, upper bound: 13.8052437
time: 27.97 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 67.98 seconds
IS_A1_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.6880222, upper bound: 13.8046101
IS_A1_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7098236, upper bound: 13.8046801
IS_A1_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7091130, upper bound: 13.8046101
IS_A1_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7308952, upper bound: 13.8046801
IS_A1_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7045256, upper bound: 13.8046101
IS_A1_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7263107, upper bound: 13.8046801
IS_A1_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7256151, upper bound: 13.8046101
IS_A1_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7473756, upper bound: 13.8046801
IS_A1_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7472402, upper bound: 13.8051575
IS_A1_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7690087, upper bound: 13.8052436
IS_A2_B2_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7693271
IS_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7694222
IS_A2_B2_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7858128
IS_A2_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7858960
IS_A2_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7242655, upper bound: 13.8046101
IS_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7460642, upper bound: 13.8046801
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7453554, upper bound: 13.8046101
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7671341, upper bound: 13.8046801
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7407686, upper bound: 13.8046101
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7625503, upper bound: 13.8046801
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7618571, upper bound: 13.8046101
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7836123, upper bound: 13.8046801
IS_A2_B2_B2_A2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7675997
IS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7677092
IS_A2_B2_B2_A2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7834814, upper bound: 13.7886723
IS_A2_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7887686
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7623970, upper bound: 13.8051575
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7841912, upper bound: 13.8052437
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.7834814, upper bound: 13.8051575
IS_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 67.98
Output dim: 1, lower bound: -13.8052435, upper bound: 13.8052437

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -7.8786478, 25.8885460, -7.9796791, 25.8892212, -30.6666260, 30.7388763
1: -0.2519546, 26.7488480, -0.3020802, 26.7351723, -21.2498932, 21.2880821
2: -0.2996049, 25.7894516, -0.3777714, 25.7853622, -20.7661781, 20.7914734
3: -4.6487436, 22.4748249, -4.7230039, 22.4528694, -19.8364716, 19.8256569
4: -7.5287075, 22.4894085, -7.6066885, 22.4672165, -25.4797058, 25.5047379
5: -4.7372909, 24.8010273, -4.8148441, 24.7902012, -23.6779900, 23.6745491
6: -39.4073639, -4.2612114, -39.4157677, -4.2052727, -29.6994781, 29.6612244
7: -9.0812683, 23.3670845, -9.1406555, 23.3659134, -25.6366272, 25.6666412
8: -13.7584238, 19.9589043, -13.8241177, 19.9525356, -29.1496811, 29.1688576
9: -8.4856491, 22.4960365, -8.5395689, 22.5092697, -27.7377777, 27.7651596
10: -28.9114552, 17.8635025, -28.9483376, 17.9078636, -39.6184464, 39.5892944
11: -26.1718178, 6.7866340, -26.1665745, 6.8492365, -29.5299416, 29.5759697
12: -46.0579453, -8.4272795, -46.0559692, -8.3185139, -32.3104477, 32.2538910
13: -32.4851036, 13.2093306, -32.5413589, 13.2199383, -38.4779587, 38.4885788
14: -59.3623428, -2.1449146, -59.3973427, -2.0479164, -56.7661591, 56.7866516
15: -14.1461716, 18.7794800, -14.1902218, 18.7663250, -27.9102554, 27.9278717
16: -15.5679264, 22.2341728, -15.6156101, 22.2671471, -29.5406036, 29.5104065
17: -59.1078949, -6.9437532, -59.1072197, -6.9080849, -51.2391129, 51.2127838
18: -21.9988384, 16.2532291, -21.9992771, 16.3248558, -33.0125961, 32.9872437
19: -22.1026859, 5.9733028, -22.0893669, 6.0305004, -21.8650742, 21.8499603
20: -27.7802544, 0.5423779, -27.7708416, 0.6151338, -22.2320786, 22.2419052
21: -26.2577820, 7.3119082, -26.2536297, 7.4015718, -28.7551422, 28.7517509
22: -29.4056778, 5.2201414, -29.4046326, 5.2555718, -27.3676834, 27.3889465
23: -17.7724247, 12.2215528, -17.7559814, 12.2719803, -23.8978424, 23.8859673
24: -16.4337749, 13.5742350, -16.4179649, 13.6131878, -25.7672882, 25.7005692
25: -23.7467022, 8.8032627, -23.7164764, 8.8408031, -27.7692184, 27.7672806
26: -39.2540131, 3.9514003, -39.2587852, 4.0745373, -33.0963898, 33.0782471
27: -19.4524288, 14.7995214, -19.4699211, 14.8676987, -34.3201294, 34.2694435
28: -21.3145103, 11.3726864, -21.3054962, 11.4397850, -26.1566849, 26.1397705
29: -24.3396778, 6.3418994, -24.3370132, 6.3814268, -28.5146103, 28.5091858
30: -29.9663277, 5.7823505, -29.9463158, 5.8450193, -33.8603210, 33.8217545
31: -23.2940559, 7.5856304, -23.2826672, 7.6386943, -24.9242249, 24.9023018
32: -36.9695969, -2.6637611, -36.9735565, -2.6231070, -29.7547379, 29.7415619
33: -54.3781815, -0.0087585, -54.4412346, 0.0324345, -44.3555298, 44.3708115
34: -49.0996208, -7.0736217, -49.1324348, -7.0295582, -35.7759705, 35.7559891
35: -40.2499008, 3.8301382, -40.2925949, 3.8749161, -35.9254150, 35.9507523
36: -45.5406418, 1.2449026, -45.5688972, 1.2923994, -39.6005554, 39.6251297
37: -60.6401138, -9.9440212, -60.6467743, -9.9026394, -44.7960205, 44.8189163
38: -53.5721741, 3.9349957, -53.6158905, 4.0171108, -48.6747742, 48.7014771
39: -61.8597832, -4.6239614, -61.8768921, -4.6017809, -48.2043915, 48.2421265
40: -50.1439514, -9.2044764, -50.1481476, -9.1934175, -39.0561523, 39.0359421
41: -31.9692307, 7.2279749, -31.9920578, 7.2665319, -37.9205933, 37.9038391
42: -30.0796413, -0.3743467, -30.0855446, -0.3402290, -23.0412788, 23.0183296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1670

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6764153, upper bound: 13.8003623
time: 34.32 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6838215, upper bound: 13.8003623
time: 37.57 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -7.9606037, 25.9168892, -8.0144320, 25.8904057, -30.7457848, 30.8020630
1: -0.2997637, 26.7752876, -0.3238020, 26.7359352, -21.2925644, 21.3365059
2: -0.3510218, 25.8227425, -0.4009752, 25.7860451, -20.8120346, 20.8484116
3: -4.6957207, 22.5052624, -4.7451749, 22.4543362, -19.8770752, 19.8760681
4: -7.5932059, 22.5187454, -7.6358876, 22.4683418, -25.5374069, 25.5653000
5: -4.7794037, 24.8283463, -4.8341780, 24.7914047, -23.7131386, 23.7177620
6: -39.4397354, -4.2037792, -39.4177284, -4.1824303, -29.7648697, 29.7219315
7: -9.1198616, 23.3826714, -9.1580153, 23.3664646, -25.6699219, 25.7018051
8: -13.8270512, 19.9952946, -13.8558254, 19.9540100, -29.2115135, 29.2368622
9: -8.5285311, 22.5177708, -8.5573473, 22.5109329, -27.7800980, 27.8093796
10: -28.9492760, 17.8947144, -28.9639683, 17.9112091, -39.6603088, 39.6433945
11: -26.2173233, 6.8338294, -26.1682777, 6.8705649, -29.5966034, 29.6190948
12: -46.0623779, -8.3830719, -46.0568581, -8.3008652, -32.3281631, 32.2987976
13: -32.5253220, 13.2446365, -32.5587692, 13.2252541, -38.5187607, 38.5572128
14: -59.4047890, -2.1191549, -59.4138794, -2.0449524, -56.8073578, 56.8405609
15: -14.2127571, 18.8201675, -14.2192001, 18.7690811, -27.9727936, 28.0014038
16: -15.5885296, 22.2493744, -15.6215229, 22.2678833, -29.5670929, 29.5238914
17: -59.1349602, -6.9163589, -59.1176338, -6.9046516, -51.2659683, 51.2577515
18: -22.0421276, 16.2917404, -22.0035629, 16.3422680, -33.0801849, 33.0276489
19: -22.1526432, 6.0204630, -22.0918465, 6.0528312, -21.9372635, 21.8919106
20: -27.8390484, 0.5981083, -27.7728500, 0.6410036, -22.3171387, 22.2895851
21: -26.3283958, 7.3802004, -26.2568970, 7.4335122, -28.8576431, 28.8139496
22: -29.4499702, 5.2538772, -29.4062996, 5.2703743, -27.4311447, 27.4180107
23: -17.8168602, 12.2638588, -17.7579670, 12.2910748, -23.9614792, 23.9215584
24: -16.4826813, 13.6130877, -16.4202957, 13.6314621, -25.8346786, 25.7360611
25: -23.7955093, 8.8597174, -23.7182693, 8.8658829, -27.8435211, 27.8163033
26: -39.2999306, 3.9966035, -39.2613029, 4.0959387, -33.1680145, 33.1165466
27: -19.5044441, 14.8483009, -19.4740887, 14.8902359, -34.3946800, 34.3223877
28: -21.3620110, 11.4231195, -21.3077412, 11.4633389, -26.2276306, 26.1847038
29: -24.3805389, 6.3711710, -24.3382988, 6.3937111, -28.5672379, 28.5349808
30: -30.0260773, 5.8474040, -29.9479828, 5.8741879, -33.9481506, 33.8823395
31: -23.3451080, 7.6427937, -23.2848167, 7.6650534, -25.0018921, 24.9543152
32: -36.9871368, -2.6325669, -36.9759636, -2.6107607, -29.7863998, 29.7735748
33: -54.4180946, 0.0257978, -54.4454384, 0.0473661, -44.4111023, 44.4028625
34: -49.1301346, -7.0522280, -49.1356583, -7.0206404, -35.8172684, 35.7833862
35: -40.2803802, 3.8544788, -40.2961044, 3.8857689, -35.9737701, 35.9828033
36: -45.5792046, 1.2817039, -45.5709457, 1.3090553, -39.6568985, 39.6616592
37: -60.6907768, -9.9140263, -60.6509399, -9.8887587, -44.8554840, 44.8427811
38: -53.6292534, 3.9909277, -53.6198082, 4.0412235, -48.7572784, 48.7511902
39: -61.8971710, -4.6029797, -61.8811569, -4.5935478, -48.2525635, 48.2641068
40: -50.1688766, -9.1955814, -50.1529999, -9.1913319, -39.0910034, 39.0526505
41: -31.9898949, 7.2548337, -31.9949837, 7.2773743, -37.9610901, 37.9359512
42: -30.0899963, -0.3463774, -30.0871506, -0.3288298, -23.0662842, 23.0500908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1670

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6982076, upper bound: 13.8004298
time: 34.18 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7056177, upper bound: 13.8004298
time: 33.87 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -7.8843117, 25.8996925, -8.0177345, 25.9117889, -30.6865997, 30.7877388
1: -0.2544589, 26.7653370, -0.3341532, 26.7687263, -21.2625999, 21.3387680
2: -0.3027065, 25.8036537, -0.4028444, 25.8138638, -20.7794495, 20.8326111
3: -4.6507263, 22.4932308, -4.7499447, 22.4894562, -19.8453140, 19.8726540
4: -7.5326729, 22.5126266, -7.6443362, 22.5140648, -25.5090561, 25.5696259
5: -4.7424150, 24.8170967, -4.8454337, 24.8219986, -23.6980515, 23.7241821
6: -39.4097366, -4.2572222, -39.4262314, -4.1942148, -29.7101288, 29.6782837
7: -9.0869827, 23.3822803, -9.1752338, 23.3967495, -25.6510735, 25.7179642
8: -13.7600784, 19.9744282, -13.8530655, 19.9857368, -29.1568031, 29.2113991
9: -8.4883080, 22.5089302, -8.5581799, 22.5350151, -27.7548828, 27.7993240
10: -28.9139633, 17.8731804, -28.9665833, 17.9286041, -39.6289062, 39.6229477
11: -26.1847229, 6.7881575, -26.1988697, 6.8703418, -29.5661545, 29.5961304
12: -46.0639877, -8.4221077, -46.0699425, -8.2952042, -32.3290863, 32.2686119
13: -32.4969711, 13.2125645, -32.5672798, 13.2371502, -38.4918518, 38.5082245
14: -59.3725166, -2.1433544, -59.4227600, -2.0417233, -56.7807465, 56.8077698
15: -14.1491756, 18.7946625, -14.2171211, 18.7981186, -27.9358292, 27.9721909
16: -15.5717926, 22.2461929, -15.6367321, 22.2897129, -29.5444183, 29.5467262
17: -59.1277504, -6.9414330, -59.1574326, -6.8797359, -51.2888336, 51.2358704
18: -22.0038319, 16.2554207, -22.0210495, 16.3309116, -33.0197830, 33.0105438
19: -22.1242561, 5.9737067, -22.1346283, 6.0514097, -21.9068527, 21.8626900
20: -27.7995682, 0.5431623, -27.8103580, 0.6324253, -22.2729874, 22.2627716
21: -26.2753849, 7.3133755, -26.2918301, 7.4209847, -28.7900696, 28.7630997
22: -29.4217091, 5.2222919, -29.4392509, 5.2740059, -27.3988800, 27.3955231
23: -17.7930489, 12.2241535, -17.7987423, 12.2990856, -23.9497452, 23.9117317
24: -16.4518013, 13.5751896, -16.4549751, 13.6292934, -25.8047104, 25.7256699
25: -23.7818718, 8.8070831, -23.7873688, 8.8782825, -27.8466568, 27.8045044
26: -39.2684746, 3.9545794, -39.2897415, 4.0933051, -33.1322784, 33.0955276
27: -19.4564133, 14.8010902, -19.4789257, 14.8728819, -34.3292961, 34.2800140
28: -21.3373299, 11.3746490, -21.3519478, 11.4661922, -26.2099609, 26.1730118
29: -24.3503342, 6.3434882, -24.3619041, 6.3946939, -28.5331726, 28.5217209
30: -29.9821358, 5.7848115, -29.9795303, 5.8684382, -33.9013519, 33.8509521
31: -23.3148785, 7.5866632, -23.3271809, 7.6643181, -24.9738617, 24.9172211
32: -36.9723663, -2.6599946, -36.9833336, -2.6121283, -29.7677460, 29.7566833
33: -54.3905296, -0.0063801, -54.4666977, 0.0521469, -44.3857422, 44.3755569
34: -49.1013107, -7.0696983, -49.1417847, -7.0177269, -35.7907562, 35.7665024
35: -40.2576485, 3.8315449, -40.3088188, 3.8842106, -35.9482651, 35.9518356
36: -45.5560760, 1.2464848, -45.5997810, 1.3131933, -39.6247559, 39.6320496
37: -60.6605682, -9.9419880, -60.6896782, -9.8799915, -44.8286591, 44.8303986
38: -53.5853577, 3.9380140, -53.6445694, 4.0328770, -48.6961975, 48.7161255
39: -61.8801155, -4.6208506, -61.9179192, -4.5744076, -48.2426300, 48.2534790
40: -50.1470642, -9.2019215, -50.1625290, -9.1834354, -39.0654907, 39.0560226
41: -31.9720421, 7.2385602, -32.0061874, 7.2895999, -37.9433594, 37.9287262
42: -30.0886936, -0.3699903, -30.1044502, -0.3216572, -23.0680542, 23.0378838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1670

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6975020, upper bound: 13.8003623
time: 37.17 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7049149, upper bound: 13.8003623
time: 37.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.9662066, 25.9280376, -8.0524712, 25.9129696, -30.7657318, 30.8509369
1: -0.3022842, 26.7917595, -0.3558793, 26.7695065, -21.3052559, 21.3871880
2: -0.3541265, 25.8369370, -0.4260607, 25.8145523, -20.8253021, 20.8895302
3: -4.6976671, 22.5236664, -4.7721443, 22.4908752, -19.8859253, 19.9230881
4: -7.5971651, 22.5420074, -7.6734738, 22.5151730, -25.5667381, 25.6301994
5: -4.7845135, 24.8444004, -4.8647566, 24.8231792, -23.7331924, 23.7674026
6: -39.4420967, -4.1998158, -39.4281616, -4.1713600, -29.7754898, 29.7389908
7: -9.1255398, 23.3978615, -9.1925669, 23.3972836, -25.6843643, 25.7530975
8: -13.8286810, 20.0108223, -13.8848228, 19.9872303, -29.2186241, 29.2794151
9: -8.5311842, 22.5306721, -8.5759249, 22.5366879, -27.7971725, 27.8435745
10: -28.9517670, 17.9043713, -28.9822330, 17.9319344, -39.6707764, 39.6771240
11: -26.2302647, 6.8353767, -26.2006111, 6.8917150, -29.6327705, 29.6392365
12: -46.0684586, -8.3779078, -46.0707855, -8.2775974, -32.3468170, 32.3135376
13: -32.5371819, 13.2478886, -32.5846977, 13.2424202, -38.5326080, 38.5768356
14: -59.4148750, -2.1176472, -59.4392624, -2.0388041, -56.8219604, 56.8615417
15: -14.2157784, 18.8353729, -14.2461147, 18.8009224, -27.9983826, 28.0457230
16: -15.5923328, 22.2613182, -15.6425896, 22.2904034, -29.5709305, 29.5602150
17: -59.1548538, -6.9140453, -59.1677704, -6.8763514, -51.3156509, 51.2807770
18: -22.0470963, 16.2938652, -22.0253143, 16.3482513, -33.0873566, 33.0509796
19: -22.1742172, 6.0208645, -22.1371498, 6.0736990, -21.9789925, 21.9046631
20: -27.8584003, 0.5988603, -27.8123474, 0.6582394, -22.3580322, 22.3104706
21: -26.3459721, 7.3817310, -26.2950687, 7.4529548, -28.8925552, 28.8252640
22: -29.4660606, 5.2560143, -29.4409351, 5.2887363, -27.4623413, 27.4246101
23: -17.8374672, 12.2665014, -17.8006821, 12.3181515, -24.0133743, 23.9473267
24: -16.5006809, 13.6140537, -16.4573212, 13.6475477, -25.8720779, 25.7611542
25: -23.8306732, 8.8635569, -23.7891350, 8.9034195, -27.9209824, 27.8535042
26: -39.3143768, 3.9999139, -39.2922478, 4.1145954, -33.2039185, 33.1338501
27: -19.5084419, 14.8499565, -19.4831123, 14.8954096, -34.4038506, 34.3330688
28: -21.3848114, 11.4251251, -21.3541546, 11.4897299, -26.2808990, 26.2178917
29: -24.3911705, 6.3727188, -24.3632317, 6.4070148, -28.5858154, 28.5475159
30: -30.0418835, 5.8498621, -29.9812107, 5.8976922, -33.9891968, 33.9115906
31: -23.3659306, 7.6438284, -23.3293571, 7.6906948, -25.0514984, 24.9692535
32: -36.9899063, -2.6287551, -36.9857330, -2.5997696, -29.7993851, 29.7887497
33: -54.4305229, 0.0281868, -54.4708862, 0.0669918, -44.4412842, 44.4076538
34: -49.1318893, -7.0482712, -49.1450157, -7.0088158, -35.8320999, 35.7939529
35: -40.2881355, 3.8559895, -40.3123398, 3.8950253, -35.9966202, 35.9838333
36: -45.5946617, 1.2833118, -45.6018753, 1.3297653, -39.6811905, 39.6686096
37: -60.7112885, -9.9119692, -60.6938858, -9.8661461, -44.8881531, 44.8541870
38: -53.6423798, 3.9939585, -53.6484032, 4.0569935, -48.7788086, 48.7658920
39: -61.9174805, -4.5999146, -61.9221687, -4.5662117, -48.2907715, 48.2753677
40: -50.1720009, -9.1930618, -50.1673584, -9.1813641, -39.1003265, 39.0726929
41: -31.9926529, 7.2653732, -32.0090866, 7.3005066, -37.9838409, 37.9608078
42: -30.0990677, -0.3420858, -30.1060410, -0.3102293, -23.0930557, 23.0696259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=188, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1497
type: B, layer: 1, pos: 1497
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: A, layer: 1, pos: 1419
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1670

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7192575, upper bound: 13.8004298
time: 40.29 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7266824, upper bound: 13.8004298
time: 31.39 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 74.03 seconds
IS_A1_B2_B2_A1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.6764153, upper bound: 13.8003623
IS_A1_B2_B2_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.6838215, upper bound: 13.8003623
IS_A1_B2_B2_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.6982076, upper bound: 13.8004298
IS_A1_B2_B2_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.7056177, upper bound: 13.8004298
IS_A1_B2_B2_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.6975020, upper bound: 13.8003623
IS_A1_B2_B2_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.7049149, upper bound: 13.8003623
IS_A1_B2_B2_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.7192575, upper bound: 13.8004298
IS_A1_B2_B2_A1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 74.03
Output dim: 1, lower bound: -13.7266824, upper bound: 13.8004298
IS_A1_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7045256, upper bound: 13.8046101
IS_A1_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7263107, upper bound: 13.8046801
IS_A1_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7256151, upper bound: 13.8046101
IS_A1_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7473756, upper bound: 13.8046801
IS_A1_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7472402, upper bound: 13.8051575
IS_A1_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7690087, upper bound: 13.8052436
IS_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7694222
IS_A2_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.8052435, upper bound: 13.7858960
IS_A2_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7242655, upper bound: 13.8046101
IS_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7460642, upper bound: 13.8046801
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7453554, upper bound: 13.8046101
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7671341, upper bound: 13.8046801
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7407686, upper bound: 13.8046101
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7625503, upper bound: 13.8046801
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7618571, upper bound: 13.8046101
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7836123, upper bound: 13.8046801
IS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7677092
IS_A2_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.8052436, upper bound: 13.7887686
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7623970, upper bound: 13.8051575
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7841912, upper bound: 13.8052437
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.7834814, upper bound: 13.8051575
IS_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 74.03
Output dim: 1, lower bound: -13.8052435, upper bound: 13.8052437

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 54.54 + 3590.22 = 3644.76 seconds
