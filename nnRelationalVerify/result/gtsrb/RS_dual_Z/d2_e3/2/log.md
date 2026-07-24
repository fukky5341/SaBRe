## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.33 + 52.79 = 55.12 seconds
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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8140958, upper bound: 13.7967490
time: 33.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7967490, upper bound: 13.8140959
time: 31.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 64.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 64.58
Output dim: 1, lower bound: -13.8140958, upper bound: 13.7967490
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 64.58
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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7977102, upper bound: 13.7961873
time: 29.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8135342, upper bound: 13.7803598
time: 31.04 seconds

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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7803598, upper bound: 13.8135342
time: 32.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7961873, upper bound: 13.7977102
time: 26.05 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 60.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 60.41
Output dim: 1, lower bound: -13.7977102, upper bound: 13.7961873
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 60.41
Output dim: 1, lower bound: -13.8135342, upper bound: 13.7803598
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 60.41
Output dim: 1, lower bound: -13.7803598, upper bound: 13.8135342
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 60.41
Output dim: 1, lower bound: -13.7961873, upper bound: 13.7977102

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0366478, 31.0349846
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4946404, 21.4899635
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0289764, 21.0216560
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1182976, 20.1080475
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7812042, 25.7727890
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9913521, 23.9844284
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8361435, 29.8367004
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9209023, 25.9155502
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4213181, 29.4182129
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9880524, 27.9919968
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8116531, 39.8301849
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8569107, 29.8564453
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5467758, 32.5533485
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6067657, 38.5936890
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1378784, 57.1420135
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0969009, 28.0965919
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7051315, 29.7139053
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5328369, 51.5358276
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1652756, 33.1691895
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0862656, 22.0947227
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4965057, 22.5016327
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0520020, 29.0566673
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6186676, 27.6196899
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1435089, 24.1519012
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9372330, 25.9392242
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0589600, 28.0660477
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4686661, 33.4712677
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4343414, 26.4429092
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7355652, 28.7372780
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0951767, 34.1020279
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1487961, 25.1553574
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8908844, 29.8905945
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6006165, 44.5944519
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -36.0048294, 36.0082245
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1412430, 36.1358871
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8491516, 39.8432999
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0106354, 45.0188522
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9666595, 48.9571304
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4145813, 48.4067841
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1628418, 39.1700516
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0801315, 38.0814056
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1187172, 23.1211777

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7976594, upper bound: 13.7871548
time: 36.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7837739, upper bound: 13.7951133
time: 35.26 seconds

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

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8134844, upper bound: 13.7713452
time: 29.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7995873, upper bound: 13.7792935
time: 32.60 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7792935, upper bound: 13.7995873
time: 30.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7713452, upper bound: 13.8134845
time: 32.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0349846, 31.0366478
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4899635, 21.4946404
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0216522, 21.0289726
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1080513, 20.1183014
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7727966, 25.7811966
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9844246, 23.9913559
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8367081, 29.8361435
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9155464, 25.9208984
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4182129, 29.4213181
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9919968, 27.9880524
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8301926, 39.8116608
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8564453, 29.8569069
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5533524, 32.5467758
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5936737, 38.6067810
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1420135, 57.1378784
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0965958, 28.0969009
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7139053, 29.7051353
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5358276, 51.5328369
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1691895, 33.1652756
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0947266, 22.0862694
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.5016327, 22.4965019
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0566711, 29.0520020
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6196899, 27.6186676
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1519012, 24.1435089
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9392242, 25.9372330
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0660477, 28.0589600
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4712753, 33.4686737
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4429092, 26.4343414
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7372742, 28.7355652
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.1020279, 34.0951767
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1553574, 25.1487923
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8905945, 29.8908844
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5944519, 44.6006165
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -36.0082169, 36.0048370
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1358871, 36.1412506
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8433075, 39.8491440
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0188446, 45.0106277
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9571381, 48.9666595
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4067841, 48.4145813
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1700592, 39.1628342
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0814133, 38.0801468
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.1211739, 23.1187172

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7951133, upper bound: 13.7837739
time: 33.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7871548, upper bound: 13.7976594
time: 35.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 70.45 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7976594, upper bound: 13.7871548
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7837739, upper bound: 13.7951133
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.8134844, upper bound: 13.7713452
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7995873, upper bound: 13.7792935
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7792935, upper bound: 13.7995873
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7713452, upper bound: 13.8134845
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7951133, upper bound: 13.7837739
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 70.45
Output dim: 1, lower bound: -13.7871548, upper bound: 13.7976594

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0097847, 31.0050354
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4741096, 21.4662781
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -21.0018044, 20.9897461
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.1013718, 20.0882645
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7513008, 25.7377243
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9582291, 23.9461517
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8119659, 29.8122787
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9096947, 25.9020691
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4095001, 29.4047012
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9666595, 27.9673996
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8030014, 39.8210297
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8208618, 29.8245010
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5123291, 32.5242996
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5953445, 38.5801468
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1280518, 57.1341400
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0561142, 28.0513458
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7238617, 29.7286682
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5091400, 51.5157471
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1507874, 33.1566315
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0810776, 22.0919304
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4558945, 22.4687996
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0193253, 29.0303650
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6093597, 27.6131821
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1100082, 24.1238594
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9208450, 25.9263611
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0106964, 28.0245247
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4292145, 33.4378967
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3897858, 26.4047394
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7350922, 28.7372513
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0502014, 34.0631104
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1302261, 25.1392441
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8890533, 29.8909988
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6126862, 44.6107635
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9746246, 35.9805908
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1329346, 36.1281738
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8504791, 39.8447418
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9978485, 45.0102539
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9406433, 48.9361267
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4278259, 48.4209290
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1578522, 39.1656723
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0800476, 38.0812073
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0945663, 23.0998650

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7607937, upper bound: 13.7858804
time: 35.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7963867, upper bound: 13.7502600
time: 31.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0067024, 31.0079155
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4709587, 21.4692345
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9970589, 20.9944916
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0985184, 20.0909805
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7461357, 25.7428894
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9530792, 23.9513016
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8117218, 29.8125229
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9074211, 25.9039917
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4078064, 29.4063454
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9634552, 27.9699783
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8024979, 39.8215256
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8245316, 29.8204002
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5177307, 32.5189018
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5932388, 38.5822601
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1299896, 57.1322021
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0516510, 28.0552177
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7198944, 29.7319641
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5127563, 51.5121307
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1526794, 33.1547012
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0834732, 22.0895348
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4636688, 22.4610252
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0257034, 29.0239944
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6121597, 27.6103821
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1154709, 24.1183968
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9241409, 25.9228363
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0174332, 28.0177841
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4351196, 33.4318085
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3961639, 26.3983459
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7355118, 28.7368011
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562592, 34.0570526
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1326828, 25.1367912
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8909912, 29.8887634
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6165924, 44.6065292
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9771881, 35.9780197
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1334686, 36.1275635
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8505707, 39.8445816
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0020294, 45.0060654
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9454346, 48.9311066
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4282837, 48.4200287
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1581573, 39.1650772
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0799408, 38.0813065
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0965424, 23.0970230

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7468479, upper bound: 13.7938389
time: 72.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7824980, upper bound: 13.7582318
time: 37.09 seconds

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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7766538, upper bound: 13.7700684
time: 26.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8122145, upper bound: 13.7344148
time: 46.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0067635, 31.0078506
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4710884, 21.4691010
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9970665, 20.9944916
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0950928, 20.0944061
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7471046, 25.7419205
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9520798, 23.9523010
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8114243, 29.8128128
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9060936, 25.9053192
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4084244, 29.4057312
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9655228, 27.9679070
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8123093, 39.8117218
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8214417, 29.8234863
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5189590, 32.5176697
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5862350, 38.5892563
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1336670, 57.1285248
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0539627, 28.0529099
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7284698, 29.7233810
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5156860, 51.5092010
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1565399, 33.1508408
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0827560, 22.0902519
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4600220, 22.4646683
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0205002, 29.0291901
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6113129, 27.6112289
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1142349, 24.1196327
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9237900, 25.9231873
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0145721, 28.0206451
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4350433, 33.4318848
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3962402, 26.3982849
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7351913, 28.7371216
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562592, 34.0570526
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1305923, 25.1388741
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8893585, 29.8903961
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6098022, 44.6133194
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9805756, 35.9746399
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1293793, 36.1316605
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8453674, 39.8497925
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0019073, 45.0061874
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9361572, 48.9403992
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4205322, 48.4277802
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1631317, 39.1601028
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0806427, 38.0806122
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0943222, 23.0992470

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7627020, upper bound: 13.7780177
time: 36.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7983133, upper bound: 13.7423744
time: 33.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0078468, 31.0067673
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4690971, 21.4710922
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9944878, 20.9970665
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0944061, 20.0950890
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7419167, 25.7471085
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9523010, 23.9520798
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8128128, 29.8114243
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9053154, 25.9060974
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4057312, 29.4084206
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9679108, 27.9655228
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8117142, 39.8123093
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8234863, 29.8214417
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5176697, 32.5189590
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5892563, 38.5862427
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1285248, 57.1336670
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0529099, 28.0539627
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7233810, 29.7284737
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5091858, 51.5156860
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1508408, 33.1565399
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0902481, 22.0827560
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4646683, 22.4600258
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0291901, 29.0205002
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6112289, 27.6113129
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1196289, 24.1142311
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9231873, 25.9237900
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0206451, 28.0145721
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4318848, 33.4350433
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3982773, 26.3962402
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7371216, 28.7351913
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0570526, 34.0562592
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1388779, 25.1305923
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8903961, 29.8893585
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6133270, 44.6097946
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9746399, 35.9805832
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1316528, 36.1293793
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8497772, 39.8453674
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0061798, 45.0019073
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9403992, 48.9361496
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4277954, 48.4205322
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1600952, 39.1631241
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0806122, 38.0806427
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0992508, 23.0943184

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7423744, upper bound: 13.7983133
time: 33.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7780177, upper bound: 13.7627020
time: 34.31 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7344148, upper bound: 13.8122145
time: 29.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7700684, upper bound: 13.7766538
time: 44.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0079231, 31.0066986
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4692345, 21.4709549
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9944954, 20.9970627
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0909805, 20.0985184
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7428932, 25.7461319
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9513016, 23.9530792
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8125229, 29.8117218
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9039879, 25.9074211
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4063492, 29.4078064
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9699783, 27.9634514
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8215256, 39.8024979
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8204041, 29.8245316
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5188980, 32.5177307
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5822525, 38.5932388
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1322021, 57.1300049
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0552216, 28.0516548
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7319565, 29.7198944
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5121307, 51.5127411
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1547012, 33.1526794
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0895386, 22.0834732
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4610214, 22.4636650
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0239944, 29.0256996
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6103821, 27.6121597
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1184006, 24.1154671
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9228363, 25.9241409
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0177841, 28.0174332
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4318085, 33.4351196
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3983459, 26.3961639
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7368011, 28.7355118
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0570526, 34.0562592
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1367874, 25.1326752
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8887634, 29.8909912
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6065216, 44.6166000
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9780121, 35.9772034
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1275635, 36.1334686
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8445740, 39.8505783
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0060577, 45.0020294
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9311066, 48.9454346
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4200439, 48.4282837
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1650696, 39.1581497
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0812988, 38.0799408
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0970230, 23.0965424

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7582318, upper bound: 13.7824980
time: 28.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7938389, upper bound: 13.7468479
time: 29.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0050392, 31.0097809
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4662743, 21.4741096
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9897499, 21.0018082
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0882645, 20.1013718
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7377281, 25.7513008
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9461517, 23.9582291
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8122787, 29.8119659
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.9020729, 25.9096985
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.4047012, 29.4095001
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9673996, 27.9666634
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.8210373, 39.8029938
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8245010, 29.8208618
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5242996, 32.5123291
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5801468, 38.5953522
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1341400, 57.1280518
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0513458, 28.0561142
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.7286606, 29.7238617
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5157471, 51.5091400
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1566315, 33.1507874
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0919342, 22.0810738
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4687958, 22.4558945
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0303650, 29.0193253
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.6131821, 27.6093597
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.1238556, 24.1100044
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9263611, 25.9208450
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -28.0245285, 28.0106964
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4378967, 33.4292068
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.4047394, 26.3897858
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.7372513, 28.7350922
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0631104, 34.0502014
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1392441, 25.1302261
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.8909988, 29.8890533
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.6107635, 44.6126862
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9805908, 35.9746323
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.1281738, 36.1329346
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8447418, 39.8504868
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -45.0102539, 44.9978409
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9361267, 48.9406433
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4209290, 48.4278259
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1656647, 39.1578522
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0812073, 38.0800476
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0998611, 23.0945663

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7502600, upper bound: 13.7963867
time: 34.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7858803, upper bound: 13.7607937
time: 34.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 71.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7607937, upper bound: 13.7858804
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7963867, upper bound: 13.7502600
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7468479, upper bound: 13.7938389
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7824980, upper bound: 13.7582318
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7766538, upper bound: 13.7700684
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.8122145, upper bound: 13.7344148
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7627020, upper bound: 13.7780177
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7983133, upper bound: 13.7423744
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7423744, upper bound: 13.7983133
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7780177, upper bound: 13.7627020
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7344148, upper bound: 13.8122145
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7700684, upper bound: 13.7766538
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7582318, upper bound: 13.7824980
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7938389, upper bound: 13.7468479
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7502600, upper bound: 13.7963867
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.25
Output dim: 1, lower bound: -13.7858803, upper bound: 13.7607937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0107193, 31.0059395
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4469261, 21.4434814
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9663963, 20.9601707
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0596657, 20.0532112
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7077637, 25.7012749
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9176788, 23.9122238
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8265762, 29.8297653
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8473167, 25.8499222
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3709564, 29.3725090
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9311295, 27.9377213
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7627563, 39.7872925
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8246307, 29.8321190
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5163651, 32.5200348
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6164246, 38.5976639
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1576843, 57.1592407
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0573502, 28.0504799
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6551361, 29.6714020
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5868225, 51.5807343
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1410370, 33.1449585
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0617332, 22.0689354
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4467010, 22.4580345
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0065460, 29.0153732
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5593643, 27.5532455
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0811310, 24.0894241
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9038467, 25.9061127
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9604263, 27.9644012
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3959656, 33.3982544
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3562622, 26.3647232
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6958618, 28.6901474
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0490417, 34.0606155
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1212921, 25.1288834
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9046021, 29.9093323
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5547943, 44.5410538
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9466248, 35.9466553
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0841370, 36.0696182
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8262787, 39.8154373
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9446259, 44.9426727
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9035034, 48.8914642
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4103088, 48.3996964
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1553650, 39.1645050
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0867920, 38.0891724
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0845108, 23.0883598

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7347359, upper bound: 13.7857264
time: 44.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7606447, upper bound: 13.7598936
time: 29.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0106812, 31.0059814
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4513130, 21.4390907
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9722328, 20.9543381
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0663185, 20.0465546
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7148438, 25.6941872
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9243011, 23.9056091
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8294449, 29.8268967
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8575478, 25.8396912
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3773193, 29.3661613
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9369812, 27.9318657
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7692566, 39.7807922
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8284836, 29.8282700
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5080643, 32.5283356
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6128693, 38.6012192
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1531525, 57.1637726
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0552444, 28.0525818
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6665955, 29.6599388
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5741272, 51.5934296
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1391144, 33.1468811
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0580864, 22.0725937
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4451294, 22.4596062
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0043335, 29.0175858
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5494232, 27.5631866
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0755768, 24.0949783
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9006042, 25.9093552
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9505692, 27.9742584
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3895569, 33.4046631
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3497696, 26.3712234
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6879883, 28.6980286
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0477142, 34.0619507
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1198654, 25.1303101
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9073944, 29.9065475
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5429840, 44.5528870
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9407043, 35.9525757
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0743713, 36.0793762
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8211823, 39.8205338
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9302673, 44.9570465
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8959808, 48.8989944
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4065857, 48.4034042
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1566925, 39.1631775
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0880127, 38.0879517
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0830612, 23.0898094

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7703965, upper bound: 13.7501151
time: 37.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7962247, upper bound: 13.7242226
time: 34.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0076447, 31.0088196
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4437675, 21.4464417
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9616508, 20.9649162
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0568047, 20.0559273
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7025909, 25.7064400
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9125366, 23.9173737
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8263397, 29.8300018
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8450432, 25.8518410
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3692627, 29.3741570
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9279175, 27.9403000
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7622528, 39.7877884
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8283005, 29.8280182
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5217590, 32.5146370
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6143188, 38.5997696
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1596222, 57.1572876
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0528946, 28.0543518
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6511688, 29.6746979
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5904236, 51.5771179
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1429291, 33.1430283
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0641441, 22.0665398
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4544754, 22.4502602
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0129166, 29.0090027
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5621643, 27.5504456
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0865860, 24.0839653
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9071350, 25.9025955
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9671631, 27.9576607
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4018860, 33.3921661
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3626556, 26.3583450
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6962891, 28.6896973
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0551147, 34.0545578
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1237411, 25.1264305
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9065399, 29.9071045
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5587311, 44.5368195
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9491882, 35.9440842
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0846863, 36.0690079
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8263702, 39.8152771
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9488220, 44.9384918
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9083099, 48.8864441
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4107666, 48.3988113
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1556549, 39.1639099
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0866852, 38.0892715
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0864868, 23.0855179

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7207648, upper bound: 13.7936839
time: 29.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7466993, upper bound: 13.7678578
time: 33.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0075989, 31.0088577
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4481621, 21.4420471
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9674873, 20.9590836
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0634651, 20.0492706
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7096863, 25.6993523
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9191513, 23.9107590
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8292007, 29.8271408
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8552742, 25.8416138
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3756104, 29.3678055
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9337692, 27.9344444
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7687683, 39.7812881
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8321457, 29.8241653
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5134659, 32.5229340
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6107635, 38.6033325
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1550903, 57.1618347
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0507889, 28.0564537
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6626282, 29.6632347
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5777283, 51.5898285
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1410065, 33.1449509
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0604820, 22.0701981
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4529037, 22.4518318
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0107040, 29.0112152
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5522232, 27.5603905
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0810318, 24.0895195
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9038925, 25.9058380
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9573059, 27.9675179
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3954773, 33.3985748
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3561554, 26.3648376
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6884003, 28.6975746
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0537720, 34.0558929
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1223145, 25.1278610
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9093246, 29.9043121
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5468903, 44.5486450
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9432678, 35.9500122
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0749207, 36.0787735
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8212738, 39.8203659
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9344482, 44.9528580
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9007721, 48.8939743
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4070435, 48.4025116
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1569824, 39.1625748
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0879059, 38.0880585
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0850372, 23.0869675

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7564695, upper bound: 13.7580854
time: 42.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7823424, upper bound: 13.7321964
time: 33.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0107880, 31.0058746
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4470634, 21.4433479
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9663963, 20.9601669
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0562325, 20.0566406
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7087402, 25.7002983
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9166870, 23.9132271
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8262863, 29.8300552
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8459892, 25.8512459
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3715820, 29.3718948
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9332047, 27.9356461
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7725677, 39.7774811
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8215408, 29.8352051
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5175934, 32.5188065
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6094208, 38.6046600
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1613464, 57.1555634
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0596619, 28.0481682
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6637115, 29.6628227
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5897522, 51.5778046
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1449051, 33.1410980
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0610313, 22.0696526
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4430542, 22.4616776
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0013504, 29.0205727
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5585251, 27.5540924
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0798950, 24.0906601
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9034958, 25.9064636
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9575653, 27.9672623
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3959045, 33.3983307
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3563385, 26.3646622
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6955414, 28.6904678
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0490417, 34.0606155
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1192093, 25.1309662
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9029770, 29.9109650
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5480194, 44.5478516
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9500122, 35.9432678
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0800476, 36.0737076
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8210754, 39.8206558
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9445038, 44.9427948
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8942108, 48.9007568
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4025574, 48.4074478
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1603394, 39.1595306
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0874939, 38.0884705
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0822830, 23.0905800

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7506322, upper bound: 13.7699141
time: 29.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7765066, upper bound: 13.7440367
time: 37.82 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7862707, upper bound: 13.7342662
time: 48.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8120514, upper bound: 13.7083323
time: 28.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0077057, 31.0087509
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4439049, 21.4463043
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9616508, 20.9649124
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0533791, 20.0593567
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7035675, 25.7054672
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9115295, 23.9183769
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8260422, 29.8302994
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8437157, 25.8531647
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3698883, 29.3735390
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9299927, 27.9382248
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7720642, 39.7779846
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8252106, 29.8311081
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5229950, 32.5134048
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6073151, 38.6067734
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1632996, 57.1536255
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0551987, 28.0520439
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6597443, 29.6661186
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5933838, 51.5741730
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1467896, 33.1391602
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0634270, 22.0672569
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4508286, 22.4539032
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0077209, 29.0141983
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5613174, 27.5512924
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0853577, 24.0851974
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9067917, 25.9029388
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9643097, 27.9605217
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4018097, 33.3922424
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3627167, 26.3582687
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6959686, 28.6900177
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0551147, 34.0545578
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1216583, 25.1285133
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9049072, 29.9087372
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5519257, 44.5436096
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9525757, 35.9407043
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0805817, 36.0731049
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8211670, 39.8204880
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9487000, 44.9386139
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8990173, 48.8957367
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4030151, 48.4065552
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1606293, 39.1589279
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0873871, 38.0885773
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0842667, 23.0877419

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7366651, upper bound: 13.7778615
time: 28.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7625558, upper bound: 13.7519942
time: 35.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0076675, 31.0087891
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4482918, 21.4419136
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9674950, 20.9590759
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0600395, 20.0527000
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7106628, 25.6983795
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9181519, 23.9117584
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8289108, 29.8274307
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8539467, 25.8429413
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3762360, 29.3671913
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9358444, 27.9323730
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7785797, 39.7714767
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8290634, 29.8272552
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5146942, 32.5217056
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6037598, 38.6103287
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1587524, 57.1581573
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0531006, 28.0541458
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6712036, 29.6546555
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5806580, 51.5868683
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1448669, 33.1410904
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0597649, 22.0709152
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4492645, 22.4554710
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0055084, 29.0164108
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5513763, 27.5612373
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0798035, 24.0907516
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9035492, 25.9061813
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9544525, 27.9703789
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3954010, 33.3986511
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3562317, 26.3647690
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6880798, 28.6978951
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0537720, 34.0558929
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1202316, 25.1299438
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9076996, 29.9059448
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5400848, 44.5554428
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9466553, 35.9466248
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0708313, 36.0828629
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8160706, 39.8255844
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9343262, 44.9529800
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8914795, 48.9032593
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3993225, 48.4102554
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1619568, 39.1576004
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0886078, 38.0873566
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0828171, 23.0891914

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7723314, upper bound: 13.7422259
time: 32.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7981592, upper bound: 13.7162973
time: 32.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0087891, 31.0076675
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4419136, 21.4482956
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9590797, 20.9674911
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0527000, 20.0600357
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6983795, 25.7106552
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9117584, 23.9181519
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8274307, 29.8289108
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8429375, 25.8539429
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3671875, 29.3762283
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9323730, 27.9358444
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7714691, 39.7785721
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8272552, 29.8290596
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5217056, 32.5146942
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6103363, 38.6037521
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1581573, 57.1587524
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0541458, 28.0531006
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6546555, 29.6712074
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5868835, 51.5806732
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1410904, 33.1448669
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0709190, 22.0597610
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4554749, 22.4492607
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0164108, 29.0055084
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5612411, 27.5513763
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0907516, 24.0797997
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9061813, 25.9035492
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9703827, 27.9544487
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3986511, 33.3954010
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3647614, 26.3562317
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6978912, 28.6880836
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0558929, 34.0537720
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1299438, 25.1202316
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9059448, 29.9076996
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5554352, 44.5400848
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9466248, 35.9466400
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0828705, 36.0708237
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8255920, 39.8160629
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9529877, 44.9343262
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9032593, 48.8914795
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4102478, 48.3993073
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1576080, 39.1619644
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0873566, 38.0886078
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0891876, 23.0828133

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7162973, upper bound: 13.7981592
time: 31.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7422259, upper bound: 13.7723314
time: 28.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0087509, 31.0077095
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4463081, 21.4439049
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9649162, 20.9616547
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0593529, 20.0533791
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7054749, 25.7035675
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9183807, 23.9115334
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8302994, 29.8260422
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8531685, 25.8437157
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3735352, 29.3698807
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9382248, 27.9299927
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7779846, 39.7720642
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8311081, 29.8252106
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5134048, 32.5229950
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6067657, 38.6073151
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1536255, 57.1632996
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0520401, 28.0552025
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6661224, 29.6597443
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5741882, 51.5933685
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1391602, 33.1467896
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0672569, 22.0634232
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4539032, 22.4508324
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0141983, 29.0077209
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5512924, 27.5613174
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0851974, 24.0853577
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9029388, 25.9067917
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9605255, 27.9643059
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3922424, 33.4018097
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3582687, 26.3627243
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6900177, 28.6959648
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0545654, 34.0550995
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1285172, 25.1216583
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9087296, 29.9049072
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5436249, 44.5519257
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9407043, 35.9525681
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0731049, 36.0805817
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8204956, 39.8211594
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9386139, 44.9487000
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8957367, 48.8990173
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4065552, 48.4030075
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1589355, 39.1606369
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0885773, 38.0873871
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0877457, 23.0842628

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7519942, upper bound: 13.7625558
time: 42.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7778615, upper bound: 13.7366651
time: 30.85 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7083323, upper bound: 13.8120514
time: 35.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7342663, upper bound: 13.7862707
time: 32.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0058746, 31.0107880
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4433479, 21.4470596
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9601707, 20.9664001
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0566368, 20.0562363
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7003021, 25.7087326
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9132233, 23.9166832
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8300552, 29.8262863
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8512459, 25.8459930
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3719025, 29.3715782
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9356461, 27.9332047
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7774811, 39.7725677
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8352051, 29.8215408
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5188065, 32.5175934
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6046600, 38.6094284
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1555634, 57.1613617
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0481644, 28.0596619
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6628265, 29.6637115
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5777893, 51.5897675
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1410980, 33.1449051
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0696526, 22.0610275
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4616776, 22.4430580
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0205765, 29.0013504
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5540924, 27.5585213
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0906601, 24.0798950
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9064560, 25.9034958
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9672623, 27.9575653
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3983154, 33.3959045
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3646545, 26.3563385
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6904678, 28.6955452
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0606232, 34.0490417
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1309662, 25.1192093
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9109650, 29.9029694
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5478668, 44.5480118
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9432678, 35.9500046
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0737152, 36.0800476
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8206482, 39.8210678
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9427948, 44.9445114
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.9007568, 48.8942184
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4074402, 48.4025497
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1595306, 39.1603317
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0884705, 38.0874863
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0905838, 23.0822868

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7083323, upper bound: 13.7765066
time: 42.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7699141, upper bound: 13.7506322
time: 33.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0088577, 31.0076027
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4420509, 21.4481583
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9590797, 20.9674873
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0492668, 20.0634613
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6993561, 25.7096825
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9107590, 23.9191513
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8271408, 29.8292007
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8416100, 25.8552704
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3678131, 29.3756142
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9344482, 27.9337692
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7812805, 39.7687607
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8241653, 29.8321457
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5229340, 32.5134659
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6033325, 38.6107559
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1618347, 57.1550903
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0564499, 28.0507889
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6632385, 29.6626282
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5898132, 51.5777435
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1449509, 33.1410065
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0702019, 22.0604782
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4518280, 22.4529037
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0112152, 29.0107079
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5603867, 27.5522232
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0895233, 24.0810318
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9058380, 25.9038925
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9675140, 27.9573097
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3985748, 33.3954773
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3648376, 26.3561630
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6975708, 28.6884041
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0558929, 34.0537720
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1278610, 25.1223145
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9043121, 29.9093323
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5486603, 44.5468903
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9500122, 35.9432602
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0787659, 36.0749130
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8203735, 39.8212814
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9528503, 44.9344482
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8939667, 48.9007721
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4024963, 48.4070587
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1625824, 39.1569901
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0880585, 38.0879059
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0869675, 23.0850372

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7321964, upper bound: 13.7823424
time: 31.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7580854, upper bound: 13.7564695
time: 35.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0088196, 31.0076447
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4464455, 21.4437714
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9649162, 20.9616508
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0559273, 20.0568085
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7064362, 25.7025948
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9173737, 23.9125366
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8300018, 29.8263397
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8518410, 25.8450394
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3741608, 29.3692665
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9403000, 27.9279175
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7877960, 39.7622604
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8280182, 29.8282967
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5146332, 32.5217628
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5997772, 38.6143112
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1572876, 57.1596222
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0543518, 28.0528908
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6746979, 29.6511688
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5771179, 51.5904388
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1430283, 33.1429291
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0665398, 22.0641403
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4502640, 22.4544716
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0090027, 29.0129204
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5504456, 27.5621643
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0839615, 24.0865898
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9025955, 25.9071350
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9576569, 27.9671669
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3921661, 33.4018860
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3583374, 26.3626556
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6896973, 28.6962852
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0545654, 34.0550995
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1264267, 25.1237450
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9071045, 29.9065399
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5368195, 44.5587158
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9440918, 35.9491882
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0690155, 36.0846786
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8152771, 39.8263702
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9384918, 44.9488220
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8864441, 48.9083023
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3988037, 48.4107590
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1639099, 39.1556549
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0892639, 38.0866928
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0855179, 23.0864830

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7678578, upper bound: 13.7466993
time: 37.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7936839, upper bound: 13.7207648
time: 34.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0059814, 31.0106812
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4390907, 21.4513168
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9543343, 20.9722328
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0465584, 20.0663185
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6941833, 25.7148476
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9056091, 23.9243011
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8268967, 29.8294449
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8396950, 25.8575439
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3661652, 29.3773117
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9318695, 27.9369850
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7807922, 39.7692566
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8282700, 29.8284798
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5283356, 32.5080643
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6012268, 38.6128693
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1637726, 57.1531525
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0525818, 28.0552483
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6599426, 29.6665955
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5934448, 51.5741119
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1468811, 33.1391144
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0725975, 22.0580826
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4596024, 22.4451294
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0175858, 29.0043335
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5631866, 27.5494232
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0949783, 24.0755730
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9093552, 25.9006042
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9742584, 27.9505692
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.4046631, 33.3895645
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3712158, 26.3497696
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6980286, 28.6879845
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0619507, 34.0477142
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1303101, 25.1198654
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9065475, 29.9073944
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5529022, 44.5429764
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9525757, 35.9406967
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0793762, 36.0743790
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8205261, 39.8211899
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9570465, 44.9302673
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8990021, 48.8959808
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.4034119, 48.4065933
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1631775, 39.1566849
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0879517, 38.0880051
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0898056, 23.0830612

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7242226, upper bound: 13.7962247
time: 42.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7501151, upper bound: 13.7703965
time: 34.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -31.0059357, 31.0107193
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4434853, 21.4469261
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9601707, 20.9663963
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0532112, 20.0596619
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7012787, 25.7077599
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9122238, 23.9176826
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8297653, 29.8265762
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8499260, 25.8473206
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3725128, 29.3709602
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9377213, 27.9311295
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7872925, 39.7627563
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8321152, 29.8246307
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5200348, 32.5163651
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.5976562, 38.6164246
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1592255, 57.1576843
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0504837, 28.0573502
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6714020, 29.6551361
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5807190, 51.5868073
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1449585, 33.1410370
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0689354, 22.0617409
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4580383, 22.4466972
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0153732, 29.0065460
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5532455, 27.5593681
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0894241, 24.0811272
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9061127, 25.9038467
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9644012, 27.9604263
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3982544, 33.3959808
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3647232, 26.3562698
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6901474, 28.6958656
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0606232, 34.0490417
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1288834, 25.1212921
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9093323, 29.9046021
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5410614, 44.5548096
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9466553, 35.9466248
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0696106, 36.0841446
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8154297, 39.8262787
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9426727, 44.9446335
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8914642, 48.9035034
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3996887, 48.4103012
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1645050, 39.1553574
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0891724, 38.0867920
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0883560, 23.0845070

Time for backsubstitution: 1.78 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7598935, upper bound: 13.7606447
time: 36.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7857264, upper bound: 13.7347359
time: 30.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 69.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7347359, upper bound: 13.7857264
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7606447, upper bound: 13.7598936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7703965, upper bound: 13.7501151
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7962247, upper bound: 13.7242226
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7207648, upper bound: 13.7936839
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7466993, upper bound: 13.7678578
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7564695, upper bound: 13.7580854
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7823424, upper bound: 13.7321964
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7506322, upper bound: 13.7699141
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7765066, upper bound: 13.7440367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7862707, upper bound: 13.7342662
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.8120514, upper bound: 13.7083323
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7366651, upper bound: 13.7778615
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7625558, upper bound: 13.7519942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7723314, upper bound: 13.7422259
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7981592, upper bound: 13.7162973
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7162973, upper bound: 13.7981592
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7422259, upper bound: 13.7723314
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7519942, upper bound: 13.7625558
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7778615, upper bound: 13.7366651
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7083323, upper bound: 13.8120514
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7342663, upper bound: 13.7862707
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7083323, upper bound: 13.7765066
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7699141, upper bound: 13.7506322
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7321964, upper bound: 13.7823424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7580854, upper bound: 13.7564695
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7678578, upper bound: 13.7466993
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7936839, upper bound: 13.7207648
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7242226, upper bound: 13.7962247
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7501151, upper bound: 13.7703965
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7598935, upper bound: 13.7606447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.78
Output dim: 1, lower bound: -13.7857264, upper bound: 13.7347359

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9967728, 30.9934654
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4326515, 21.4313126
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9549828, 20.9505310
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0599365, 20.0535278
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7043877, 25.6993370
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9071922, 23.9034805
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8300171, 29.8337936
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8268623, 25.8325615
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3579330, 29.3615875
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9139252, 27.9232712
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7421722, 39.7700577
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8218994, 29.8297272
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5190430, 32.5201302
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6350250, 38.6147308
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1537628, 57.1553955
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0565262, 28.0477142
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6069489, 29.6310577
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6046829, 51.5961609
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1434250, 33.1481552
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0624695, 22.0695534
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4417915, 22.4523087
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0058289, 29.0144844
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5378952, 27.5275230
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0790634, 24.0869827
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8997955, 25.9013443
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9509583, 27.9531250
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3806763, 33.3804474
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3488922, 26.3560410
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6871796, 28.6783180
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0575714, 34.0678635
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1292953, 25.1364441
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9021759, 29.9070129
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5204773, 44.5004272
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9203949, 35.9153137
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0556412, 36.0354309
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8097000, 39.7971802
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9251404, 44.9196930
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8873749, 48.8743515
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3983002, 48.3858337
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1579742, 39.1676025
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0900574, 38.0927124
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0835037, 23.0873299

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7212822, upper bound: 13.7850856
time: 32.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7325550, upper bound: 13.7590943
time: 38.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9982529, 30.9919891
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4347496, 21.4292107
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9567604, 20.9487610
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0599747, 20.0534821
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7058220, 25.6978951
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9089394, 23.9017334
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8306122, 29.8331985
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8299599, 25.8294640
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3600464, 29.3594742
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9166794, 27.9205170
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7455139, 39.7667160
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8222351, 29.8293915
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5164566, 32.5227165
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6334991, 38.6162567
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1538391, 57.1553192
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0545883, 28.0496521
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6147919, 29.6232147
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6022568, 51.5986023
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1442413, 33.1473389
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0623627, 22.0696678
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4409676, 22.4531479
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0056610, 29.0146561
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5336456, 27.5317726
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0786819, 24.0873566
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8990784, 25.9020615
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9491501, 27.9549332
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3781738, 33.3829422
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3475723, 26.3573532
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6840363, 28.6814575
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562897, 34.0691757
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1288528, 25.1368790
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9022751, 29.9069061
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5141907, 44.5067139
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9152679, 35.9204254
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0499496, 36.0411148
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8080215, 39.7988510
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9216614, 44.9231873
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8863983, 48.8753357
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3964386, 48.3877106
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1584625, 39.1671066
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0903320, 38.0924377
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0834732, 23.0873642

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7467107, upper bound: 13.7592910
time: 37.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7584306, upper bound: 13.7356730
time: 35.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9967346, 30.9935074
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4370461, 21.4269180
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9608192, 20.9446945
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0665894, 20.0468674
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7114677, 25.6922493
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9138069, 23.8968582
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8328857, 29.8309250
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8370934, 25.8223343
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3642807, 29.3552399
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9197769, 27.9174194
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7486877, 39.7635498
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8257523, 29.8258781
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5107422, 32.5284271
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6314545, 38.6182861
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1492310, 57.1599274
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0544205, 28.0498161
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6184082, 29.6195984
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5919876, 51.6088715
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1415024, 33.1500854
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0588074, 22.0732117
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4402351, 22.4538803
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0036163, 29.0167007
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5279541, 27.5374680
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0735016, 24.0925369
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8965530, 25.9045868
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9411011, 27.9629822
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3742676, 33.3868637
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3423996, 26.3625336
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6792984, 28.6861992
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562592, 34.0691986
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1278687, 25.1378708
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9049606, 29.9042206
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5086365, 44.5122604
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9144745, 35.9212341
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0458908, 36.0451965
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8046036, 39.8022690
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9107666, 44.9340591
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8798523, 48.8818817
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3946075, 48.3895416
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1592865, 39.1662750
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0912781, 38.0914917
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0820618, 23.0887756

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7461685, upper bound: 13.7479027
time: 33.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7697928, upper bound: 13.7361969
time: 30.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9982147, 30.9920273
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4391441, 21.4248199
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9625969, 20.9429245
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0666351, 20.0468254
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7129173, 25.6908073
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9155540, 23.8951149
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8334808, 29.8303375
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8401909, 25.8192368
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3663940, 29.3531265
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9225388, 27.9146614
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7520142, 39.7602081
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8260956, 29.8255386
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5081558, 32.5310173
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6299286, 38.6198120
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1493073, 57.1598511
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0524826, 28.0517540
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6262512, 29.6117516
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5895615, 51.6112976
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1423187, 33.1492691
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0587006, 22.0733223
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4393959, 22.4546967
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0034485, 29.0168724
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5237045, 27.5417175
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0731277, 24.0929108
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8958359, 25.9053040
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9392929, 27.9647903
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3717651, 33.3893585
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3410873, 26.3638458
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6761551, 28.6893387
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0549622, 34.0704727
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1274261, 25.1383057
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9050598, 29.9041138
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5023499, 44.5185547
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9093475, 35.9263458
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0401993, 36.0508804
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8029251, 39.8039474
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9072876, 44.9375534
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8788605, 48.8828659
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3927307, 48.3914108
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1597748, 39.1657791
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0915527, 38.0912247
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0820236, 23.0888100

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7696007, upper bound: 13.7220423
time: 33.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7955853, upper bound: 13.7107660
time: 32.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9936905, 30.9963455
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4295006, 21.4342690
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9502373, 20.9552765
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0570755, 20.0562401
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6992149, 25.7045021
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9020424, 23.9086304
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8297806, 29.8340378
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8245811, 25.8344803
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3562317, 29.3632355
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9107132, 27.9258537
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7416840, 39.7705536
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8255692, 29.8256302
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5244446, 32.5147285
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6329041, 38.6168365
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1557007, 57.1534576
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0520706, 28.0515900
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6029816, 29.6343536
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6082993, 51.5925598
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1453094, 33.1462326
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0648651, 22.0671577
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4495583, 22.4445343
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0122070, 29.0081100
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5406952, 27.5247269
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0845184, 24.0815201
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9030838, 25.8978271
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9577026, 27.9463844
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3865814, 33.3743668
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3552780, 26.3496552
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6875992, 28.6778679
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0636292, 34.0618057
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1317444, 25.1339912
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9041138, 29.9047775
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5243835, 44.4961853
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9229584, 35.9127350
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0561752, 36.0348282
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8097916, 39.7970123
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9293213, 44.9155045
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8921814, 48.8693237
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3987732, 48.3849487
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1582489, 39.1670074
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0899658, 38.0928116
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0854874, 23.0844879

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7072943, upper bound: 13.7930433
time: 29.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7185817, upper bound: 13.7670597
time: 27.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9951706, 30.9948654
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4315987, 21.4321671
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9520149, 20.9535065
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0571213, 20.0561981
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7006645, 25.7030602
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9037895, 23.9068832
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8303680, 29.8334427
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8276863, 25.8313828
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3583450, 29.3611221
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9134674, 27.9230957
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7450256, 39.7672119
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8259048, 29.8252869
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5218506, 32.5173187
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6313782, 38.6183701
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1557922, 57.1533813
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0501251, 28.0535278
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6108246, 29.6265106
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6058731, 51.5949707
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1461334, 33.1454086
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0647583, 22.0672722
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4487495, 22.4453735
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0120316, 29.0082855
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5364456, 27.5289764
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0841446, 24.0818977
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9023666, 25.8985443
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9558868, 27.9481964
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3840942, 33.3768616
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3539658, 26.3509674
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6844559, 28.6810074
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0623474, 34.0631180
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1313019, 25.1344299
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9042130, 29.9046783
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5180969, 44.5024796
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9178467, 35.9178543
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0504990, 36.0405121
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8081131, 39.7986908
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9258423, 44.9189987
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8911896, 48.8703156
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3968964, 48.3868103
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1587677, 39.1665115
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0902252, 38.0925446
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0854568, 23.0845184

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7327528, upper bound: 13.7672552
time: 29.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7444830, upper bound: 13.7436379
time: 32.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9936523, 30.9963837
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4338875, 21.4298782
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9560738, 20.9494400
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0637360, 20.0495834
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7063103, 25.6974144
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9086647, 23.9020081
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8326416, 29.8311691
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8348122, 25.8242569
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3625870, 29.3568878
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9165649, 27.9199982
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7481842, 39.7640533
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8294220, 29.8217773
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5161438, 32.5230293
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6293488, 38.6203995
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1511688, 57.1579895
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0499649, 28.0536919
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6144409, 29.6228905
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5956039, 51.6052551
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1433868, 33.1481552
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0612106, 22.0708160
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4480171, 22.4461060
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0099945, 29.0103226
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5307541, 27.5346680
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0789642, 24.0870781
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8998413, 25.9010696
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9478455, 27.9562416
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3801727, 33.3807755
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3487778, 26.3561478
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6797180, 28.6857452
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0623322, 34.0631409
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1303177, 25.1354218
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9068985, 29.9019852
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5125427, 44.5080261
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9170380, 35.9186630
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0464096, 36.0445938
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8046951, 39.8021088
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9149780, 44.9298706
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8846436, 48.8768616
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3950653, 48.3886490
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1595917, 39.1656799
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0911713, 38.0915985
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0840378, 23.0859375

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7322081, upper bound: 13.7558722
time: 30.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7558651, upper bound: 13.7441641
time: 31.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9951324, 30.9949036
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4359856, 21.4277802
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9578514, 20.9476700
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0637741, 20.0495415
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7077446, 25.6959724
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9104042, 23.9002647
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8332367, 29.8305740
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8379097, 25.8211555
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3646927, 29.3547707
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9193268, 27.9172401
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7515259, 39.7607117
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8297577, 29.8214378
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5135574, 32.5256157
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6278229, 38.6219254
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1512451, 57.1579132
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0480270, 28.0556297
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6222839, 29.6150475
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5931778, 51.6076813
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1442108, 33.1473312
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0610962, 22.0709267
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4471779, 22.4469223
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0098190, 29.0104980
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5265045, 27.5389175
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0785904, 24.0874519
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8991241, 25.9017868
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9460297, 27.9580536
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3776703, 33.3832703
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3474655, 26.3574600
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6765747, 28.6888885
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0610199, 34.0644150
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1298752, 25.1358566
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9069977, 29.9018860
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5062561, 44.5143127
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9119110, 35.9237823
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0407333, 36.0502777
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8030167, 39.8037872
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9114685, 44.9333649
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8836517, 48.8778458
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3931885, 48.3905182
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1600800, 39.1651840
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0914459, 38.0913239
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0840073, 23.0859680

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7556705, upper bound: 13.7300179
time: 34.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7817015, upper bound: 13.7187535
time: 37.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9968414, 30.9934006
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4327888, 21.4311752
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9549904, 20.9505310
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0565033, 20.0569534
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7053642, 25.6983643
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9061928, 23.9044800
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8297272, 29.8340912
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8255348, 25.8338890
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3585434, 29.3609772
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9160004, 27.9211998
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7519836, 39.7602463
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8188171, 29.8328171
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5202789, 32.5188980
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6280212, 38.6217270
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1574402, 57.1517181
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0588303, 28.0454063
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6155243, 29.6224823
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6076279, 51.5932312
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1472931, 33.1442947
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0617523, 22.0702705
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4381447, 22.4559517
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0006332, 29.0196838
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5370483, 27.5283699
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0778275, 24.0882149
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8994522, 25.9016876
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9481049, 27.9559860
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3805847, 33.3805237
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3489609, 26.3559647
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6868591, 28.6786385
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0575714, 34.0678635
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1272125, 25.1385269
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9005432, 29.9086380
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5136719, 44.5072250
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9237823, 35.9119263
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0515518, 36.0395279
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8044815, 39.8023911
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9250183, 44.9198151
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8780823, 48.8836365
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3905640, 48.3935852
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1629486, 39.1626282
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0907593, 38.0920181
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0812912, 23.0895538

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7372074, upper bound: 13.7692726
time: 46.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7484565, upper bound: 13.7432375
time: 30.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9983215, 30.9919205
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4348869, 21.4290771
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9567604, 20.9487572
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0565491, 20.0569077
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7067986, 25.6969223
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9079323, 23.9027328
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8303146, 29.8334961
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8286324, 25.8307877
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3606567, 29.3588600
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9187546, 27.9184418
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7553253, 39.7569122
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8191528, 29.8324776
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5176849, 32.5214844
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6264954, 38.6232605
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1575165, 57.1516418
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0568924, 28.0473442
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6233673, 29.6146355
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6052017, 51.5956573
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1481018, 33.1434784
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0616455, 22.0703812
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4373360, 22.4567909
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0004578, 29.0198555
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5327988, 27.5326195
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0774536, 24.0885925
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8987274, 25.9024124
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9462891, 27.9577942
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3780975, 33.3830185
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3476486, 26.3572769
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6837158, 28.6817780
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562897, 34.0691681
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1267700, 25.1389618
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9006424, 29.9085388
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5073853, 44.5135193
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9186554, 35.9170380
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0458603, 36.0452118
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8028030, 39.8040695
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9215393, 44.9233093
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8771057, 48.8846283
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3886871, 48.3954544
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1634369, 39.1621323
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0910339, 38.0917435
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0812607, 23.0895844

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7625975, upper bound: 13.7434325
time: 31.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7742974, upper bound: 13.7197765
time: 34.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9967957, 30.9934387
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4371834, 21.4267807
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9608269, 20.9446945
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0631638, 20.0502968
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7124443, 25.6912766
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9128075, 23.8978615
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8325882, 29.8312225
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8357658, 25.8236618
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3648911, 29.3546257
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9218521, 27.9153442
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7584839, 39.7537460
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8226624, 29.8289642
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5119781, 32.5271988
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6244659, 38.6252899
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1528931, 57.1562653
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0567322, 28.0475082
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6269913, 29.6110153
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5949173, 51.6059265
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1453629, 33.1462173
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0580978, 22.0739288
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4366035, 22.4575195
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -28.9984207, 29.0218964
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5271072, 27.5383148
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0722733, 24.0937729
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8962097, 25.9049377
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9382477, 27.9658432
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3741760, 33.3869324
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3424683, 26.3624649
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6789780, 28.6865196
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0562592, 34.0691986
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1257782, 25.1399536
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9033356, 29.9058533
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5018311, 44.5190582
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9178467, 35.9178543
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0417862, 36.0492935
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7993851, 39.8074799
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9106445, 44.9341812
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8705597, 48.8911743
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3868561, 48.3972931
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1642609, 39.1613007
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0919647, 38.0908051
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0798416, 23.0909996

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7620696, upper bound: 13.7320499
time: 31.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7856694, upper bound: 13.7203206
time: 25.49 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7854751, upper bound: 13.7061498
time: 37.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8114132, upper bound: 13.6948650
time: 26.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9937592, 30.9962769
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4296379, 21.4341354
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9502449, 20.9552765
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0536499, 20.0596695
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7001915, 25.7035294
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9010429, 23.9096298
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8294830, 29.8343277
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8232613, 25.8358078
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3568497, 29.3626213
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9127884, 27.9237823
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7514954, 39.7607422
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8224792, 29.8287163
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5256729, 32.5134964
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6259155, 38.6238403
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1593781, 57.1497803
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0543747, 28.0492783
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6115570, 29.6257782
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6112442, 51.5896149
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1491776, 33.1423645
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0641479, 22.0678749
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4459267, 22.4481773
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0070038, 29.0133095
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5398483, 27.5255737
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0832901, 24.0827560
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9027405, 25.8981705
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9548416, 27.9492455
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3865051, 33.3744431
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3553467, 26.3495865
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6872787, 28.6781883
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0636292, 34.0618057
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1296616, 25.1360741
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9024811, 29.9064026
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5175781, 44.5029831
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9263458, 35.9093552
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0520859, 36.0389175
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8045731, 39.8022308
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9291992, 44.9156265
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8828888, 48.8786163
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3910217, 48.3926926
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1632233, 39.1620331
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0906525, 38.0921173
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0832596, 23.0867081

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7232187, upper bound: 13.7772206
time: 28.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7344856, upper bound: 13.7511954
time: 31.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9952393, 30.9947968
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4317284, 21.4320335
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9520149, 20.9535027
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0536957, 20.0596237
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7016258, 25.7020874
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9027901, 23.9078827
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8300781, 29.8337402
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8263588, 25.8327103
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3589630, 29.3605042
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9155426, 27.9210205
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7548370, 39.7574005
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8228149, 29.8283768
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5230865, 32.5160828
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6243744, 38.6253662
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1594543, 57.1497040
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0524368, 28.0512161
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6194000, 29.6179314
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6088028, 51.5920410
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1499939, 33.1415482
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0640411, 22.0679855
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4451027, 22.4490166
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0068359, 29.0134811
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5355988, 27.5298233
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0829163, 24.0831299
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9020233, 25.8988876
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9530334, 27.9510574
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3840027, 33.3769379
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3540344, 26.3508987
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6841354, 28.6813278
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0623474, 34.0631180
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1292267, 25.1365128
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9025803, 29.9063034
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5112915, 44.5092773
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9212189, 35.9144669
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0464096, 36.0446091
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8028946, 39.8039017
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9257202, 44.9191208
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8818970, 48.8796082
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3891449, 48.3945618
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1637421, 39.1615372
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0909271, 38.0918427
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0832291, 23.0867462

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7486356, upper bound: 13.7513898
time: 29.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7603430, upper bound: 13.7277401
time: 29.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9937210, 30.9963150
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4340248, 21.4297409
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9560814, 20.9494400
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0603104, 20.0530128
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7072716, 25.6964417
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9076576, 23.9030113
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8323517, 29.8314667
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8334846, 25.8255806
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3631973, 29.3562698
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9186401, 27.9179230
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7579956, 39.7542419
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8263321, 29.8248672
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5173798, 32.5217972
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6223602, 38.6273956
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1548462, 57.1543274
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0522766, 28.0513802
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6230240, 29.6143112
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5985336, 51.6023102
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1472473, 33.1442947
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0604935, 22.0715332
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4443703, 22.4497452
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0047913, 29.0155258
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5299072, 27.5355148
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0777283, 24.0883102
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8994980, 25.9014130
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9449844, 27.9591026
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3800964, 33.3808517
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3488541, 26.3560715
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6793976, 28.6860657
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0623322, 34.0631409
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1282349, 25.1375046
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9052734, 29.9036179
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5057678, 44.5148163
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9204102, 35.9152832
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0423203, 36.0486832
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7994766, 39.8073196
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9148560, 44.9299927
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8753510, 48.8861465
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3873138, 48.3963928
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1645660, 39.1607056
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0918732, 38.0908966
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0818100, 23.0881577

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7481096, upper bound: 13.7400094
time: 29.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7717288, upper bound: 13.7282831
time: 27.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9951935, 30.9948387
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4361229, 21.4276428
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9578514, 20.9476662
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0603485, 20.0529709
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7087212, 25.6950035
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9094048, 23.9012680
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8329391, 29.8308716
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8365822, 25.8224831
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3653107, 29.3541565
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9213943, 27.9151688
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7613373, 39.7509003
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8266678, 29.8245239
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5147858, 32.5243835
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6208191, 38.6289215
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1549225, 57.1542358
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0503387, 28.0533180
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6308670, 29.6064682
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5961075, 51.6047363
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1480713, 33.1434708
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0603790, 22.0716438
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4435310, 22.4505615
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0046234, 29.0156975
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5256500, 27.5397644
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0773544, 24.0886841
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8987808, 25.9021301
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9431763, 27.9609146
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3775940, 33.3833466
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3475418, 26.3573914
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6762543, 28.6892090
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0610199, 34.0644150
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1277924, 25.1379395
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9053726, 29.9035187
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.4994507, 44.5211105
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9152985, 35.9203949
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0366440, 36.0543747
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.7977982, 39.8089981
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9113464, 44.9334869
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8743744, 48.8871307
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3854523, 48.3982697
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1650543, 39.1602097
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0921478, 38.0906219
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0817795, 23.0881920

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7715327, upper bound: 13.7141152
time: 32.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7975185, upper bound: 13.7028338
time: 31.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9948425, 30.9951935
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4276390, 21.4361229
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9476662, 20.9578514
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0529709, 20.0603485
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6950035, 25.7087212
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9012642, 23.9094048
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8308716, 29.8329391
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8224831, 25.8365860
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3541565, 29.3653107
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9151688, 27.9213982
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7509003, 39.7613373
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8245239, 29.8266678
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5243835, 32.5147896
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6289215, 38.6208191
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1542358, 57.1549225
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0533218, 28.0503387
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6064682, 29.6308670
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6047440, 51.5960999
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1434784, 33.1480637
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0716476, 22.0603790
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4505653, 22.4435349
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0157013, 29.0046196
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5397644, 27.5256538
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0886841, 24.0773582
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9021301, 25.8987808
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9609146, 27.9431725
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3833466, 33.3776016
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3573837, 26.3475418
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6892090, 28.6762543
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0644073, 34.0610123
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1379395, 25.1277924
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9035187, 29.9053726
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5211182, 44.4994659
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9203949, 35.9152985
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0543747, 36.0366364
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8089981, 39.7978058
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9335022, 44.9113464
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8871307, 48.8743668
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3982697, 48.3854446
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1602020, 39.1650620
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0906219, 38.0921478
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0881882, 23.0817833

Time for backsubstitution: 1.78 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7028338, upper bound: 13.7975185
time: 31.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7141152, upper bound: 13.7715327
time: 50.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9963226, 30.9937172
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4297447, 21.4340248
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9494362, 20.9560776
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0530167, 20.0603027
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.6964378, 25.7072792
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9030113, 23.9076614
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8314667, 29.8323517
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8255806, 25.8334885
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3562698, 29.3631935
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9179230, 27.9186401
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7542419, 39.7579956
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8248672, 29.8263321
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5217972, 32.5173759
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6273956, 38.6223526
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1543121, 57.1548462
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0513840, 28.0522766
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6143112, 29.6230202
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.6023178, 51.5985413
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1442871, 33.1472549
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0715332, 22.0604897
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4497414, 22.4443741
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0155258, 29.0047913
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5355148, 27.5299034
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0883102, 24.0777359
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.9014130, 25.8994980
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9591064, 27.9449844
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3808594, 33.3800964
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3560715, 26.3488541
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6860657, 28.6793976
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0631409, 34.0623245
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1375046, 25.1282349
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9036179, 29.9052734
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5148315, 44.5057526
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9152832, 35.9204102
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0486832, 36.0423203
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8073196, 39.7994766
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9299927, 44.9148407
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8861542, 48.8753510
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3963928, 48.3873138
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1607208, 39.1645660
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0908966, 38.0918732
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0881577, 23.0818176

Time for backsubstitution: 1.73 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7282831, upper bound: 13.7717288
time: 34.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7400094, upper bound: 13.7481096
time: 41.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9947968, 30.9952354
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4320335, 21.4317322
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9535027, 20.9520149
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0596237, 20.0536957
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7020836, 25.7016335
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9078865, 23.9027863
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8337402, 29.8300705
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8327141, 25.8263550
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3605118, 29.3589592
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9210205, 27.9155426
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7574005, 39.7548370
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8283768, 29.8228188
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5160828, 32.5230904
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6253662, 38.6243820
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1497040, 57.1594543
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0512161, 28.0524368
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6179276, 29.6194000
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5920486, 51.6088104
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1415482, 33.1499939
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0679855, 22.0640373
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4490089, 22.4451065
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0134811, 29.0068321
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5298233, 27.5355988
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0831299, 24.0829124
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8988876, 25.9020233
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9510574, 27.9530296
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3769379, 33.3840103
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3508987, 26.3540344
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6813278, 28.6841354
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0631256, 34.0623474
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1365128, 25.1292229
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9063034, 29.9025803
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5092773, 44.5112915
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9144745, 35.9212265
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0446091, 36.0464020
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8039017, 39.8028946
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9191284, 44.9257126
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8796082, 48.8818970
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3945618, 48.3891449
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1615448, 39.1637344
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0918427, 38.0909271
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0867386, 23.0832291

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7277401, upper bound: 13.7603430
time: 32.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7513898, upper bound: 13.7486356
time: 33.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2071838, 25.9418163, -8.2071838, 25.9418163, -30.9962769, 30.9937592
1: -0.4560823, 26.8079872, -0.4560823, 26.8079872, -21.4341316, 21.4296341
2: -0.5389414, 25.8468781, -0.5389414, 25.8468781, -20.9552727, 20.9502411
3: -4.8848801, 22.5333481, -4.8848801, 22.5333481, -20.0596695, 20.0536499
4: -7.7944593, 22.5544548, -7.7944593, 22.5544548, -25.7035332, 25.7001915
5: -4.9888325, 24.8665218, -4.9888325, 24.8665218, -23.9096260, 23.9010391
6: -39.4418030, -4.1016617, -39.4418030, -4.1016617, -29.8343277, 29.8294830
7: -9.3116970, 23.4362946, -9.3116970, 23.4362946, -25.8358116, 25.8232574
8: -13.9925299, 20.0270424, -13.9925299, 20.0270424, -29.3626251, 29.3568459
9: -8.6510458, 22.5643425, -8.6510458, 22.5643425, -27.9237823, 27.9127884
10: -29.0540237, 17.9963531, -29.0540237, 17.9963531, -39.7607422, 39.7514954
11: -26.2552547, 6.9770675, -26.2552547, 6.9770675, -29.8287201, 29.8224792
12: -46.1048470, -8.1314478, -46.1048470, -8.1314478, -32.5134964, 32.5256767
13: -32.6619949, 13.2778568, -32.6619949, 13.2778568, -38.6238403, 38.6259079
14: -59.5632210, -1.9373569, -59.5632210, -1.9373569, -57.1497803, 57.1593781
15: -14.3263149, 18.8376007, -14.3263149, 18.8376007, -28.0492783, 28.0543747
16: -15.7218628, 22.3303528, -15.7218628, 22.3303528, -29.6257782, 29.6115570
17: -59.2632484, -6.7908068, -59.2632484, -6.7908068, -51.5896072, 51.6112366
18: -22.0715408, 16.4265099, -22.0715408, 16.4265099, -33.1423645, 33.1491776
19: -22.1879139, 6.1556244, -22.1879139, 6.1556244, -22.0678711, 22.0641518
20: -27.8611336, 0.7479396, -27.8611336, 0.7479396, -22.4481697, 22.4459229
21: -26.3426590, 7.5490837, -26.3426590, 7.5490837, -29.0133057, 29.0070038
22: -29.5043697, 5.3699164, -29.5043697, 5.3699164, -27.5255737, 27.5398483
23: -17.8525810, 12.4018326, -17.8525810, 12.4018326, -24.0827560, 24.0832901
24: -16.5099030, 13.7176571, -16.5099030, 13.7176571, -25.8981705, 25.9027405
25: -23.8470249, 8.9913769, -23.8470249, 8.9913769, -27.9492493, 27.9548416
26: -39.3527718, 4.2661605, -39.3527718, 4.2661605, -33.3744507, 33.3865051
27: -19.5331459, 14.9696922, -19.5331459, 14.9696922, -34.5028381, 34.5028381
28: -21.4108677, 11.5927601, -21.4108677, 11.5927601, -26.3495865, 26.3553467
29: -24.4246655, 6.4744110, -24.4246655, 6.4744110, -28.6781921, 28.6872787
30: -30.0314560, 5.9842300, -30.0314560, 5.9842300, -34.0618134, 34.0636215
31: -23.3847847, 7.7739458, -23.3847847, 7.7739458, -25.1360779, 25.1296616
32: -37.0129318, -2.5412283, -37.0129318, -2.5412283, -29.9064026, 29.9024811
33: -54.5347290, 0.1338167, -54.5347290, 0.1338167, -44.5029907, 44.5175858
34: -49.1851196, -6.9402003, -49.1851196, -6.9402003, -35.9093475, 35.9263382
35: -40.3457184, 3.9521503, -40.3457184, 3.9521503, -36.0389175, 36.0520859
36: -45.6347961, 1.4077120, -45.6347961, 1.4077120, -39.8022232, 39.8045731
37: -60.7441139, -9.7851562, -60.7441139, -9.7851562, -44.9156189, 44.9292068
38: -53.6892548, 4.1566525, -53.6892548, 4.1566525, -48.8786163, 48.8828888
39: -61.9840393, -4.5029726, -61.9840393, -4.5029726, -48.3927002, 48.3910217
40: -50.2071228, -9.1446772, -50.2071228, -9.1446772, -39.1620331, 39.1632385
41: -32.0269890, 7.3602514, -32.0269890, 7.3602514, -38.0921173, 38.0906525
42: -30.1251965, -0.2483382, -30.1251965, -0.2483382, -23.0867081, 23.0832634

Time for backsubstitution: 1.78 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7511954, upper bound: 13.7344856
time: 28.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7772206, upper bound: 13.7232187
time: 34.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 64.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7212822, upper bound: 13.7850856
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7325550, upper bound: 13.7590943
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7467107, upper bound: 13.7592910
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7584306, upper bound: 13.7356730
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7461685, upper bound: 13.7479027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7697928, upper bound: 13.7361969
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7696007, upper bound: 13.7220423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7955853, upper bound: 13.7107660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7072943, upper bound: 13.7930433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7185817, upper bound: 13.7670597
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7327528, upper bound: 13.7672552
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7444830, upper bound: 13.7436379
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7322081, upper bound: 13.7558722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7558651, upper bound: 13.7441641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7556705, upper bound: 13.7300179
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7817015, upper bound: 13.7187535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7372074, upper bound: 13.7692726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7484565, upper bound: 13.7432375
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7625975, upper bound: 13.7434325
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7742974, upper bound: 13.7197765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7620696, upper bound: 13.7320499
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7856694, upper bound: 13.7203206
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7854751, upper bound: 13.7061498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.8114132, upper bound: 13.6948650
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7232187, upper bound: 13.7772206
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7344856, upper bound: 13.7511954
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7486356, upper bound: 13.7513898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7603430, upper bound: 13.7277401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7481096, upper bound: 13.7400094
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7717288, upper bound: 13.7282831
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7715327, upper bound: 13.7141152
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7975185, upper bound: 13.7028338
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7028338, upper bound: 13.7975185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7141152, upper bound: 13.7715327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7282831, upper bound: 13.7717288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7400094, upper bound: 13.7481096
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7277401, upper bound: 13.7603430
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7513898, upper bound: 13.7486356
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7511954, upper bound: 13.7344856
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.74
Output dim: 1, lower bound: -13.7772206, upper bound: 13.7232187
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7083323, upper bound: 13.8120514
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7342663, upper bound: 13.7862707
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7083323, upper bound: 13.7765066
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7699141, upper bound: 13.7506322
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7321964, upper bound: 13.7823424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7580854, upper bound: 13.7564695
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7678578, upper bound: 13.7466993
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7936839, upper bound: 13.7207648
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7242226, upper bound: 13.7962247
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7501151, upper bound: 13.7703965
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7598935, upper bound: 13.7606447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.74
Output dim: 1, lower bound: -13.7857264, upper bound: 13.7347359

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 55.12 + 3571.96 = 3627.08 seconds
