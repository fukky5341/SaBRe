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
execution time: IAR + RelationalAnalysis = 2.94 + 51.27 = 54.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -13.8153362, upper bound: 13.8153362

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8140958, upper bound: 13.7967490
time: 32.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7967490, upper bound: 13.8140959
time: 30.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 62.95 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 62.95
Output dim: 1, lower bound: -13.8140958, upper bound: 13.7967490
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 62.95
Output dim: 1, lower bound: -13.7967490, upper bound: 13.8140959

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0356750, 31.0339432
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4958115, 21.4909935
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0313416, 21.0240135
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1022758, 20.0954552
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7889557, 25.7795715
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9774666, 23.9715424
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8409195, 29.8417740
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9043159, 25.9002914
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4262009, 29.4224739
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9904633, 27.9923401
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8803635, 39.8890839
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8380432, 29.8406677
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5610046, 32.5663452
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6529007, 38.6468124
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1692047, 57.1696777
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1146545, 28.1120377
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7539215, 29.7541161
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5581665, 51.5582275
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1962051, 33.1962509
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0916824, 22.1008606
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4950256, 22.5038071
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0296097, 29.0394745
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6168365, 27.6187057
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1516724, 24.1612930
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9364090, 25.9387436
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0421371, 28.0520897
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4690323, 33.4717102
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4319000, 26.4404030
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7326431, 28.7346725
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0951538, 34.1019897
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1545944, 25.1632423
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9000549, 29.9013901
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6389008, 44.6395416
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9851837, 35.9851913
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1617737, 36.1604919
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8782120, 39.8775940
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0012512, 45.0095978
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0175934, 49.0173416
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4564819, 48.4564285
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1337891, 39.1360321
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0725708, 38.0731354
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1350021, 23.1396904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977102, upper bound: 13.7961873
time: 28.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8135342, upper bound: 13.7803598
time: 30.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0339432, 31.0356750
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4909973, 21.4958076
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0240173, 21.0313339
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0954552, 20.1022758
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7795715, 25.7889557
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9715385, 23.9774666
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8417740, 29.8409195
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9002876, 25.9043121
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4224777, 29.4261932
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9923401, 27.9904633
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8890762, 39.8803635
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8406677, 29.8380432
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5663452, 32.5610046
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6468124, 38.6529083
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1696777, 57.1692047
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.1120377, 28.1146584
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7541122, 29.7539215
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5582275, 51.5581818
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1962509, 33.1962051
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.1008682, 22.0916862
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5037994, 22.4950333
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0394745, 29.0296059
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6187057, 27.6168365
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1612930, 24.1516685
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9387436, 25.9364090
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0520859, 28.0421371
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4717178, 33.4690323
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4403992, 26.4319038
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7346725, 28.7326431
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1019897, 34.0951462
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1632462, 25.1545944
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9013977, 29.9000549
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6395416, 44.6389084
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9851990, 35.9851837
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1604919, 36.1617661
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8775864, 39.8782196
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0095978, 45.0012512
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -49.0173492, 49.0175858
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4564209, 48.4564667
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1360168, 39.1337891
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0731354, 38.0725632
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1396866, 23.1350060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7803598, upper bound: 13.8135342
time: 31.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7961873, upper bound: 13.7977102
time: 25.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 59.53 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 59.53
Output dim: 1, lower bound: -13.7977102, upper bound: 13.7961873
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 59.53
Output dim: 1, lower bound: -13.8135342, upper bound: 13.7803598
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 59.53
Output dim: 1, lower bound: -13.7803598, upper bound: 13.8135342
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 59.53
Output dim: 1, lower bound: -13.7961873, upper bound: 13.7977102

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0367165, 31.0349197
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4947777, 21.4898262
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0289764, 21.0216522
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1148720, 20.1114769
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7821655, 25.7718163
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9903526, 23.9854279
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8358536, 29.8369980
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9195747, 25.9168777
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4219360, 29.4175987
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9901276, 27.9899254
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8214645, 39.8203735
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8538208, 29.8595314
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5480118, 32.5521164
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5997772, 38.6006851
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1415405, 57.1383514
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0992126, 28.0942802
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7137146, 29.7053299
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5357666, 51.5328979
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1691360, 33.1653214
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0855484, 22.0954399
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4928589, 22.5052757
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0467987, 29.0618668
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6178207, 27.6205368
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1422806, 24.1531334
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9368820, 25.9395676
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0560989, 28.0689087
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4685898, 33.4713440
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4344101, 26.4428406
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7352448, 28.7375984
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0951767, 34.1020279
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1467056, 25.1574402
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8892593, 29.8922272
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5938110, 44.6012497
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -36.0082169, 36.0048447
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1371536, 36.1399765
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8439331, 39.8485184
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0105133, 45.0189743
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9573822, 48.9664154
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4068298, 48.4145355
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1678162, 39.1650772
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0808334, 38.0807114
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1164894, 23.1234016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8134844, upper bound: 13.7713452
time: 28.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7995873, upper bound: 13.7792935
time: 31.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0349159, 31.0367126
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4898262, 21.4947777
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0216522, 21.0289726
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1114769, 20.1148758
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7718201, 25.7821693
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9854317, 23.9903526
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8369980, 29.8358459
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9168739, 25.9195709
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4176025, 29.4219360
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9899292, 27.9901237
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8203812, 39.8214645
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8595352, 29.8538208
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5521164, 32.5480118
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6006927, 38.5997772
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1383514, 57.1415405
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0942764, 28.0992126
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7053299, 29.7137146
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5328979, 51.5357666
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1653214, 33.1691360
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0954437, 22.0855522
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5052795, 22.4928589
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0618668, 29.0467987
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6205368, 27.6178207
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1531372, 24.1422768
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9395676, 25.9368820
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0689087, 28.0560989
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4713364, 33.4685974
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4428406, 26.4344101
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7375946, 28.7352448
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1020279, 34.0951767
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1574402, 25.1467094
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8922272, 29.8892517
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6012421, 44.5938187
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -36.0048294, 36.0082169
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1399765, 36.1371536
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8485107, 39.8439255
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0189667, 45.0105057
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9664154, 48.9573746
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4145355, 48.4068298
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1650848, 39.1678085
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0807114, 38.0808411
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1234016, 23.1164970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7792935, upper bound: 13.7995873
time: 30.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7713452, upper bound: 13.8134845
time: 32.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 64.91 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 64.91
Output dim: 1, lower bound: -13.8134844, upper bound: 13.7713452
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 64.91
Output dim: 1, lower bound: -13.7995873, upper bound: 13.7792935
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 64.91
Output dim: 1, lower bound: -13.7792935, upper bound: 13.7995873
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 64.91
Output dim: 1, lower bound: -13.7713452, upper bound: 13.8134845

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0098457, 31.0049706
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4742470, 21.4661407
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0018120, 20.9897461
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0979462, 20.0916939
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7522774, 25.7367516
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9572296, 23.9471512
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8116684, 29.8125763
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9083672, 25.9033966
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4101181, 29.4040833
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9687347, 27.9653244
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8128128, 39.8112183
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8177795, 29.8275909
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5135574, 32.5230713
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5883560, 38.5871429
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1317291, 57.1304779
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0584259, 28.0490341
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7324371, 29.7200851
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5120850, 51.5128021
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1546555, 33.1527634
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0803604, 22.0926476
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4522476, 22.4724388
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0141220, 29.0355644
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6085129, 27.6140289
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1087723, 24.1250916
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9205017, 25.9267044
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0078354, 28.0273857
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4291229, 33.4379654
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3898468, 26.4046631
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7347717, 28.7375717
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0502014, 34.0631104
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1281509, 25.1413269
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8874207, 29.8926315
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6058960, 44.6175613
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9780121, 35.9772110
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1288452, 36.1322632
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8452606, 39.8499527
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9977264, 45.0103760
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9313507, 48.9454193
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4200745, 48.4286804
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1628265, 39.1606979
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0807343, 38.0805130
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0923386, 23.1020851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7766538, upper bound: 13.7700684
time: 25.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8122145, upper bound: 13.7344148
time: 44.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0049782, 31.0098457
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4661369, 21.4742432
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9897423, 21.0018120
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0916901, 20.0979462
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7367516, 25.7522736
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9471512, 23.9572296
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8125763, 29.8116684
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9034004, 25.9083710
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4040833, 29.4101181
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9653244, 27.9687347
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8112259, 39.8128052
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8275909, 29.8177757
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5230713, 32.5135612
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5871506, 38.5883484
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1304779, 57.1317291
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0490341, 28.0584259
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7200851, 29.7324409
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5128174, 51.5120697
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1527634, 33.1546555
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0926437, 22.0803604
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4724426, 22.4522514
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0355606, 29.0141258
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6140289, 27.6085129
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1250916, 24.1087723
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9267044, 25.9205017
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0273819, 28.0078354
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4379730, 33.4291306
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4046631, 26.3898544
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7375717, 28.7347717
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0631104, 34.0502014
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1413193, 25.1281433
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8926239, 29.8874283
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6175537, 44.6058884
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9772186, 35.9780121
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1322632, 36.1288452
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8499603, 39.8452759
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0103760, 44.9977188
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9454193, 48.9313507
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4286804, 48.4200745
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1606903, 39.1628265
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0805054, 38.0807419
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1020889, 23.0923424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7344148, upper bound: 13.8122145
time: 29.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7700684, upper bound: 13.7766538
time: 43.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 74.55 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 74.55
Output dim: 1, lower bound: -13.7766538, upper bound: 13.7700684
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 74.55
Output dim: 1, lower bound: -13.8122145, upper bound: 13.7344148
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 74.55
Output dim: 1, lower bound: -13.7344148, upper bound: 13.8122145
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 74.55
Output dim: 1, lower bound: -13.7700684, upper bound: 13.7766538

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0107498, 31.0059128
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4514503, 21.4389572
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9722404, 20.9543304
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0628929, 20.0499840
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7158203, 25.6932144
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9233017, 23.9066086
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8291473, 29.8271866
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8562202, 25.8410187
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3779297, 29.3655434
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9390564, 27.9297943
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7790680, 39.7709808
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8253937, 29.8313560
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5092926, 32.5271034
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6058655, 38.6082230
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1568146, 57.1600952
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0575638, 28.0502701
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6751785, 29.6513596
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5770569, 51.5904999
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1429749, 33.1430206
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0573692, 22.0733109
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4414902, 22.4632454
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9991379, 29.0227852
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5485764, 27.5640335
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0743408, 24.0962143
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9002533, 25.9097061
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9477081, 27.9771194
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3894958, 33.4047394
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3498383, 26.3711472
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6876678, 28.6983490
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0477142, 34.0619507
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1177826, 25.1323929
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9057617, 29.9081726
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5361786, 44.5596848
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9440765, 35.9491959
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0702820, 36.0834732
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8159790, 39.8257523
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9301453, 44.9571686
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8866882, 48.9082794
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3988342, 48.4111557
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1616669, 39.1582031
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0886993, 38.0872574
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0808334, 23.0920296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7862707, upper bound: 13.7342662
time: 47.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8120514, upper bound: 13.7083323
time: 27.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0059128, 31.0107498
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4389534, 21.4514542
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9543343, 20.9722366
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0499840, 20.0628891
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6932220, 25.7158203
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9066086, 23.9233017
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8271866, 29.8291473
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8410225, 25.8562164
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3655396, 29.3779259
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9297943, 27.9390564
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7709808, 39.7790680
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8313522, 29.8253937
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5270996, 32.5092964
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6082153, 38.6058655
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1600952, 57.1568146
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0502701, 28.0575600
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6513596, 29.6751785
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5904846, 51.5770569
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1430206, 33.1429749
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0733147, 22.0573654
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4632492, 22.4414864
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0227890, 28.9991341
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5640335, 27.5485764
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0962143, 24.0743408
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9097061, 25.9002533
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9771194, 27.9477081
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4047394, 33.3894882
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3711472, 26.3498459
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6983490, 28.6876640
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0619507, 34.0477142
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1323929, 25.1177826
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9081802, 29.9057617
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5596771, 44.5361786
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9491882, 35.9440765
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0834656, 36.0702896
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8257446, 39.8159714
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9571686, 44.9301453
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9082794, 48.8866882
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4111633, 48.3988419
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1582031, 39.1616592
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0872498, 38.0886993
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0920334, 23.0808372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7083323, upper bound: 13.8120514
time: 34.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7342663, upper bound: 13.7862707
time: 31.47 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 68.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 68.77
Output dim: 1, lower bound: -13.7862707, upper bound: 13.7342662
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 68.77
Output dim: 1, lower bound: -13.8120514, upper bound: 13.7083323
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.77
Output dim: 1, lower bound: -13.7083323, upper bound: 13.8120514
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 68.77
Output dim: 1, lower bound: -13.7342663, upper bound: 13.7862707

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9982758, 30.9919586
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4392815, 21.4246826
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9625969, 20.9429207
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0632095, 20.0502548
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7138786, 25.6898346
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9145546, 23.8961182
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8331833, 29.8306351
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8388634, 25.8205643
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3670120, 29.3525085
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9246140, 27.9125900
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7618256, 39.7504044
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8230057, 29.8286285
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5093842, 32.5297852
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6229401, 38.6268158
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1529846, 57.1561737
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0547943, 28.0494461
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6348343, 29.6031723
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5925064, 51.6083527
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1461792, 33.1454010
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0579834, 22.0740395
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4357643, 22.4583359
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9982452, 29.0220718
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5228577, 27.5425644
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0718994, 24.0941467
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8954849, 25.9056549
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9364319, 27.9676514
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3716888, 33.3894348
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3411560, 26.3637772
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6758347, 28.6896591
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0549622, 34.0704727
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1253433, 25.1403923
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9034348, 29.9057465
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4955444, 44.5253448
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9127350, 35.9229660
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0360947, 36.0549774
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7977066, 39.8091583
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9071655, 44.9376755
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8695679, 48.8921585
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3849792, 48.3991623
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1647491, 39.1608047
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0922394, 38.0905304
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0798111, 23.0910339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7854751, upper bound: 13.7061498
time: 36.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8114132, upper bound: 13.6948650
time: 25.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9919586, 30.9982758
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4246864, 21.4392776
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9429207, 20.9625969
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0502548, 20.0632057
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6898308, 25.7138863
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8961143, 23.9145546
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8306351, 29.8331833
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8205605, 25.8388596
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3525162, 29.3670044
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9125900, 27.9246101
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7503967, 39.7618332
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8286285, 29.8230057
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5297852, 32.5093918
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6268158, 38.6229324
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1561737, 57.1529846
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0494461, 28.0547943
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6031723, 29.6348343
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6083603, 51.5924988
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1454010, 33.1461792
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0740433, 22.0579834
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4583321, 22.4357605
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0220718, 28.9982452
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5425644, 27.5228577
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0941467, 24.0718994
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9056549, 25.8954849
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9676514, 27.9364319
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3894348, 33.3716888
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3637772, 26.3411560
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6896591, 28.6758347
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0704803, 34.0549545
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1403961, 25.1253433
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9057465, 29.9034348
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5253601, 44.4955521
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9229584, 35.9127350
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0549850, 36.0361023
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8091660, 39.7977066
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9376831, 44.9071579
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8921509, 48.8695679
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3991547, 48.3849792
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1608124, 39.1647644
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0905304, 38.0922470
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0910263, 23.0798073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6948650, upper bound: 13.8114132
time: 34.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7061498, upper bound: 13.7854751
time: 32.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 68.88 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 68.88
Output dim: 1, lower bound: -13.7854751, upper bound: 13.7061498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.88
Output dim: 1, lower bound: -13.8114132, upper bound: 13.6948650
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.88
Output dim: 1, lower bound: -13.6948650, upper bound: 13.8114132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.88
Output dim: 1, lower bound: -13.7061498, upper bound: 13.7854751

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9879951, 30.9795609
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4287567, 21.4121628
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9543762, 20.9330902
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0634499, 20.0504761
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7113419, 25.6862221
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9081078, 23.8884506
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8347855, 29.8320618
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8254318, 25.8045044
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3567123, 29.3402328
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9154739, 27.9016571
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7532959, 39.7406311
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8209839, 29.8261490
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.4990768, 32.5230865
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6342926, 38.6416855
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1614075, 57.1663055
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0535202, 28.0500870
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6063766, 29.5691299
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6050110, 51.6235046
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1469955, 33.1461716
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0607376, 22.0767708
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4303932, 22.4536400
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9977036, 29.0216293
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5048294, 27.5275574
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0717773, 24.0940323
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8957367, 25.9059143
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9300690, 27.9622116
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3523788, 33.3732681
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3399658, 26.3626976
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6692352, 28.6847305
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0572128, 34.0730743
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1316833, 25.1459274
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8951721, 29.8981018
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4696503, 44.5036926
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8881989, 35.9024429
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0083313, 36.0317612
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7750702, 39.7902298
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8885193, 44.9221039
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8441620, 48.8709183
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3671570, 48.3842545
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1661682, 39.1620407
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0933685, 38.0914917
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0782700, 23.0895958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7797343, upper bound: 13.6936659
time: 34.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8102183, upper bound: 13.6630742
time: 27.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9795647, 30.9879913
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4121628, 21.4287567
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9330902, 20.9543800
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0504799, 20.0634460
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6862183, 25.7113419
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8884468, 23.9081116
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8320618, 29.8347855
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8045120, 25.8254395
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3402405, 29.3567123
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9016571, 27.9154778
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7406311, 39.7532959
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8261490, 29.8209839
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5230865, 32.4990768
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6416779, 38.6342850
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1663055, 57.1614075
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0500870, 28.0535202
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.5691299, 29.6063766
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6235046, 51.6050110
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1461716, 33.1469955
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0767746, 22.0607376
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4536324, 22.4303856
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0216293, 28.9977074
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5275574, 27.5048294
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0940247, 24.0717735
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9059143, 25.8957367
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9622116, 27.9300728
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3732834, 33.3523712
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3627014, 26.3399696
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6847305, 28.6692352
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0730820, 34.0572052
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1459274, 25.1316833
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8981018, 29.8951721
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5036926, 44.4696503
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9024506, 35.8881912
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0317688, 36.0083313
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7902222, 39.7750702
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9221191, 44.8885193
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8709259, 48.8441696
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3842621, 48.3671494
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1620483, 39.1661835
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0915070, 38.0933685
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0895920, 23.0782700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6630742, upper bound: 13.8102184
time: 33.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6936659, upper bound: 13.7797343
time: 30.56 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 66.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.50
Output dim: 1, lower bound: -13.7797343, upper bound: 13.6936659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 66.50
Output dim: 1, lower bound: -13.8102183, upper bound: 13.6630742
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 66.50
Output dim: 1, lower bound: -13.6630742, upper bound: 13.8102184
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.50
Output dim: 1, lower bound: -13.6936659, upper bound: 13.7797343

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9848671, 30.9764366
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4130592, 21.3945312
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9309235, 20.9051895
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0464058, 20.0305099
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6853905, 25.6546249
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8879623, 23.8644905
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8300323, 29.8268356
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8024139, 25.7774429
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3200073, 29.2964172
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8986053, 27.8823051
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7157288, 39.6967163
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8141708, 29.8199615
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5012894, 32.5312500
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6755905, 38.6935043
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1642761, 57.1694336
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0488663, 28.0453606
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.5582657, 29.5123825
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6195221, 51.6403656
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1621475, 33.1588745
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0508118, 22.0680885
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4118652, 22.4371529
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9817200, 29.0077972
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4694519, 27.4965324
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0556259, 24.0800705
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8847122, 25.8962097
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.8895340, 27.9274750
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3414154, 33.3633728
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3271561, 26.3514328
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6402740, 28.6602058
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0473099, 34.0641403
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1250305, 25.1398544
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8807755, 29.8849716
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4029846, 44.4473267
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8697586, 35.8855438
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.9602280, 35.9911118
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7163467, 39.7409744
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8370667, 44.8791046
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.7637024, 48.8033676
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3024597, 48.3297272
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1752319, 39.1692429
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0923080, 38.0903473
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0712128, 23.0852127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7939008, upper bound: 13.6625863
time: 31.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8096152, upper bound: 13.6465736
time: 30.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9764290, 30.9848633
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3945351, 21.4130592
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9051971, 20.9309273
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0305138, 20.0464020
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6546288, 25.6853867
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8644943, 23.8879623
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8268356, 29.8300323
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7774353, 25.8024139
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2964172, 29.3199997
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8823013, 27.8986053
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.6967163, 39.7157288
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8199615, 29.8141708
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5312500, 32.5012894
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6935043, 38.6755981
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1694336, 57.1642761
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0453568, 28.0488701
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.5123825, 29.5582657
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6403656, 51.6195068
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1588821, 33.1621552
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0680847, 22.0508156
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4371490, 22.4118576
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0077972, 28.9817200
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4965363, 27.4694519
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0800705, 24.0556221
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8962097, 25.8847122
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9274750, 27.8895340
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3633575, 33.3414154
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3514328, 26.3271484
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6602097, 28.6402740
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0641403, 34.0473099
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1398468, 25.1250305
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8849716, 29.8807831
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4473267, 44.4029846
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8855362, 35.8697586
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.9911118, 35.9602356
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7409744, 39.7163391
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8790894, 44.8370743
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8033752, 48.7637100
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3297272, 48.3024597
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1692505, 39.1752167
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0903549, 38.0923080
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0852127, 23.0712128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6465736, upper bound: 13.8096152
time: 24.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6625863, upper bound: 13.7939008
time: 30.65 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 57.93 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 57.93
Output dim: 1, lower bound: -13.7939008, upper bound: 13.6625863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 57.93
Output dim: 1, lower bound: -13.8096152, upper bound: 13.6465736
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 57.93
Output dim: 1, lower bound: -13.6465736, upper bound: 13.8096152
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 57.93
Output dim: 1, lower bound: -13.6625863, upper bound: 13.7939008

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9862442, 30.9765015
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4025650, 21.3819542
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9332886, 20.9061737
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0477371, 20.0319595
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6683083, 25.6345367
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8818016, 23.8577042
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8320160, 29.8288879
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7986755, 25.7733307
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3057632, 29.2795219
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8809052, 27.8618355
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.6650848, 39.6374817
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8212509, 29.8300629
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5044556, 32.5348167
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6372833, 38.6600266
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1598358, 57.1645508
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0361862, 28.0304985
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.4991913, 29.4432526
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6159210, 51.6362457
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1588364, 33.1548767
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0351944, 22.0543900
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.3748550, 22.4062767
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9598160, 28.9925957
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4651871, 27.4957428
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0285568, 24.0564690
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8714447, 25.8847961
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9050522, 27.9480286
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3023071, 33.3301849
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3122101, 26.3383026
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6462250, 28.6674271
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0524292, 34.0699997
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1068802, 25.1244698
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8731384, 29.8780289
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.3610535, 44.4120941
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8901978, 35.9032593
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.9383392, 35.9727554
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.6799774, 39.7106323
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8271179, 44.8703461
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.6963196, 48.7471542
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.2614899, 48.2954025
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2069702, 39.1964874
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.1051865, 38.1018372
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0521622, 23.0693474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7808789, upper bound: 13.6455334
time: 27.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085785, upper bound: 13.6174260
time: 28.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9765091, 30.9862366
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3819504, 21.4025650
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9061737, 20.9332886
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0319595, 20.0477409
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6345406, 25.6683006
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8577080, 23.8817978
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8288879, 29.8320160
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7733307, 25.7986832
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2795181, 29.3057518
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8618317, 27.8809052
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.6374817, 39.6650848
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8300629, 29.8212509
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5348206, 32.5044518
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6600189, 38.6372757
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1645508, 57.1598511
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0304947, 28.0361862
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.4432526, 29.4991951
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6362457, 51.6159363
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1548767, 33.1588364
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0543900, 22.0351982
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4062729, 22.3748512
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9925995, 28.9598160
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4957428, 27.4651871
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0564728, 24.0285606
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8847961, 25.8714447
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9480286, 27.9050484
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3301849, 33.3023071
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3383026, 26.3122101
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6674271, 28.6462212
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0699921, 34.0524292
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1244659, 25.1068802
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8780212, 29.8731461
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4120789, 44.3610687
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9032593, 35.8901901
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.9727478, 35.9383392
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7106323, 39.6799698
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8703461, 44.8271103
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.7471466, 48.6963272
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.2954102, 48.2614899
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1964874, 39.2069702
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.1018295, 38.1051941
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0693436, 23.0521584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6174260, upper bound: 13.8085785
time: 38.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6455334, upper bound: 13.7808789
time: 26.63 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 67.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 67.77
Output dim: 1, lower bound: -13.7808789, upper bound: 13.6455334
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 67.77
Output dim: 1, lower bound: -13.8085785, upper bound: 13.6174260
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 67.77
Output dim: 1, lower bound: -13.6174260, upper bound: 13.8085785
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 67.77
Output dim: 1, lower bound: -13.6455334, upper bound: 13.7808789

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9796143, 30.9698944
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3920975, 21.3707619
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9135666, 20.8833313
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0572739, 20.0418129
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6543045, 25.6170540
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8752861, 23.8503952
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8133774, 29.8085556
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7853813, 25.7581711
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2662811, 29.2331200
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8606873, 27.8390121
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.6196594, 39.5851288
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8180237, 29.8271065
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5097656, 32.5465622
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6789017, 38.7192764
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1602173, 57.1649170
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0390015, 28.0334778
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.4540367, 29.3868523
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6302185, 51.6506958
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.2073669, 33.1964874
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0518684, 22.0697632
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.3754044, 22.4068451
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9625854, 28.9954491
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4415359, 27.4754868
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0403214, 24.0677605
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8783722, 25.8917236
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.8915939, 27.9364281
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3219376, 33.3479614
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3312149, 26.3554382
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6390533, 28.6628571
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0532990, 34.0708466
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1226273, 25.1388130
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8372726, 29.8474503
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.2763977, 44.3405075
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8482895, 35.8683319
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.8632126, 35.9094391
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.5833511, 39.6281738
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.7828522, 44.8326645
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.5749817, 48.6432343
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.1536255, 48.2033463
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2182922, 39.2064972
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0962982, 38.0925140
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0551605, 23.0729294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1790

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7902535, upper bound: 13.6163883
time: 32.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8079238, upper bound: 13.6037803
time: 30.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9698944, 30.9796143
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3707581, 21.3920975
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.8833237, 20.9135590
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0418167, 20.0572739
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6170578, 25.6543083
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8503914, 23.8752823
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8085556, 29.8133774
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7581673, 25.7853851
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2331238, 29.2662735
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8390121, 27.8606873
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.5851288, 39.6196594
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8271027, 29.8180237
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5465622, 32.5097656
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.7192764, 38.6789017
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1649170, 57.1602173
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0334778, 28.0390015
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.3868523, 29.4540329
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6506958, 51.6302185
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1964874, 33.2073669
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0697670, 22.0518646
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4068527, 22.3754120
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9954453, 28.9625854
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4754868, 27.4415359
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0677643, 24.0403175
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8917236, 25.8783722
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9364243, 27.8915977
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3479538, 33.3219299
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3554382, 26.3312149
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6628571, 28.6390533
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0708466, 34.0532990
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1388168, 25.1226273
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8474426, 29.8372726
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.3405151, 44.2763977
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8683243, 35.8482895
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.9094467, 35.8632126
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.6281815, 39.5833511
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8326569, 44.7828445
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.6432343, 48.5749741
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.2033539, 48.1536255
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2064972, 39.2182922
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0925140, 38.0963058
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0729294, 23.0551567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1790

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6037803, upper bound: 13.8079238
time: 35.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6163882, upper bound: 13.7902535
time: 28.04 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 66.32 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 66.32
Output dim: 1, lower bound: -13.7902535, upper bound: 13.6163883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 66.32
Output dim: 1, lower bound: -13.8079238, upper bound: 13.6037803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 66.32
Output dim: 1, lower bound: -13.6037803, upper bound: 13.8079238
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 66.32
Output dim: 1, lower bound: -13.6163882, upper bound: 13.7902535

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9689407, 30.9571304
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3800201, 21.3564224
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9045181, 20.8724556
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0561371, 20.0404968
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6520348, 25.6134605
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8674965, 23.8410835
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8176804, 29.8124313
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7687454, 25.7382774
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2556381, 29.2202148
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8510971, 27.8275909
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.6120987, 39.5765839
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8142319, 29.8224792
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.4924011, 32.5334091
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6846924, 38.7288818
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1714020, 57.1786194
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0399704, 28.0367661
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.4237480, 29.3506126
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6453781, 51.6692657
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.2073746, 33.1964798
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0536652, 22.0715103
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.3711624, 22.4033737
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9620132, 28.9951401
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4213638, 27.4588699
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0399170, 24.0675163
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8784256, 25.8917847
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.8845291, 27.9307289
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.2993927, 33.3291473
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3296967, 26.3541756
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6390610, 28.6647110
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0541229, 34.0718079
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1272354, 25.1423798
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8302155, 29.8414841
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.2489166, 44.3175125
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8211517, 35.8456192
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.8320312, 35.8833618
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.5590668, 39.6078568
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.7606201, 44.8140717
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.5474854, 48.6202316
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.1343842, 48.1872559
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2209320, 39.2087402
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0984116, 38.0943680
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0558548, 23.0736504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7840527, upper bound: 13.6028785
time: 33.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8070323, upper bound: 13.5795794
time: 32.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9571304, 30.9689369
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3564224, 21.3800201
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.8724518, 20.9045181
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0404968, 20.0561371
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6134605, 25.6520424
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8410835, 23.8674927
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8124313, 29.8176804
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7382736, 25.7687454
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2202225, 29.2556343
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8275909, 27.8510971
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.5765915, 39.6120911
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8224792, 29.8142319
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5334091, 32.4923973
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.7288818, 38.6847076
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1786194, 57.1714020
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0367661, 28.0399704
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.3506126, 29.4237480
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6692734, 51.6453705
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1964798, 33.2073746
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0715103, 22.0536613
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4033737, 22.3711624
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9951401, 28.9620132
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4588699, 27.4213638
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0675201, 24.0399208
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8917847, 25.8784256
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9307327, 27.8845291
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3291473, 33.2994003
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3541718, 26.3296928
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6647110, 28.6390610
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0718231, 34.0541153
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1423798, 25.1272354
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8414841, 29.8302155
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.3175201, 44.2489166
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8456268, 35.8211594
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.8833618, 35.8320389
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.6078491, 39.5590668
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.8140717, 44.7606201
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.6202393, 48.5474930
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.1872559, 48.1343842
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2087402, 39.2209244
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0943832, 38.0984268
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0736465, 23.0558586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.5795794, upper bound: 13.8070323
time: 32.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6028785, upper bound: 13.7840527
time: 37.17 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 72.40 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 72.40
Output dim: 1, lower bound: -13.7840527, upper bound: 13.6028785
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 72.40
Output dim: 1, lower bound: -13.8070323, upper bound: 13.5795794
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 72.40
Output dim: 1, lower bound: -13.5795794, upper bound: 13.8070323
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 72.40
Output dim: 1, lower bound: -13.6028785, upper bound: 13.7840527

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9564819, 30.9429321
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3544769, 21.3269844
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.8859291, 20.8506775
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0481796, 20.0307732
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6292534, 25.5843735
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8579941, 23.8304672
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.7905426, 29.7883301
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7564774, 25.7241974
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.2285919, 29.1878471
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8576279, 27.8278885
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.5553894, 39.5092316
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8433380, 29.8572617
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.4689331, 32.5172882
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5866241, 38.6456985
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1920929, 57.2032166
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0226898, 28.0165100
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.3867035, 29.3025665
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6779785, 51.7081909
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1853485, 33.1712418
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0411949, 22.0617027
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.3447723, 22.3828659
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9717331, 29.0103989
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4253693, 27.4659882
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0519180, 24.0820045
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8819580, 25.8956070
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9088364, 27.9608154
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3181305, 33.3528976
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3300781, 26.3545494
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6581955, 28.6875648
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0556030, 34.0732346
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1377869, 25.1557579
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8051605, 29.8195724
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.1903534, 44.2677689
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8345413, 35.8568268
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.7836761, 35.8425140
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.4621658, 39.5252838
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.7017517, 44.7649078
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.4117432, 48.5044479
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.0364990, 48.1036911
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2286530, 39.2156372
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0932999, 38.0897980
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0296173, 23.0517273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7898472, upper bound: 13.5596966
time: 35.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7843195, upper bound: 13.5639675
time: 29.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9429321, 30.9564819
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.3269882, 21.3544769
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.8506813, 20.8859329
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0307693, 20.0481758
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.5843773, 25.6292458
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.8304672, 23.8579903
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.7883301, 29.7905426
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.7241974, 25.7564774
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.1878510, 29.2285919
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.8278885, 27.8576241
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.5092316, 39.5553894
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8572617, 29.8433380
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5172882, 32.4689331
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6457062, 38.5866318
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.2032166, 57.1920929
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0165100, 28.0226860
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.3025665, 29.3867035
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.7081909, 51.6779785
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1712494, 33.1853485
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0617027, 22.0411911
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.3828583, 22.3447685
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0103989, 28.9717331
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.4659882, 27.4253693
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0820084, 24.0519142
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8956070, 25.8819580
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9608154, 27.9088364
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3528900, 33.3181229
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3545532, 26.3300743
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6875610, 28.6581955
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0732422, 34.0555954
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1557617, 25.1377869
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8195724, 29.8051605
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.2677765, 44.1903534
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.8568192, 35.8345413
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -35.8425140, 35.7836761
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.5252914, 39.4621658
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.7649078, 44.7017517
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.5044403, 48.4117432
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.1036987, 48.0364914
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.2156219, 39.2286453
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0897903, 38.0932999
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0517273, 23.0296173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=192, inp2_unstable=192, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.5639675, upper bound: 13.7843195
time: 30.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.5596965, upper bound: 13.7898472
time: 207.72 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 240.83 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 240.83
Output dim: 1, lower bound: -13.7898472, upper bound: 13.5596966
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 240.83
Output dim: 1, lower bound: -13.7843195, upper bound: 13.5639675
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 240.83
Output dim: 1, lower bound: -13.5639675, upper bound: 13.7843195
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 240.83
Output dim: 1, lower bound: -13.5596965, upper bound: 13.7898472

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 54.21 + 1698.61 = 1752.82 seconds
