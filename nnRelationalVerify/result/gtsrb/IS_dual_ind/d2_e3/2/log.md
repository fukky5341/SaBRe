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
execution time: IAR + RelationalAnalysis = 2.33 + 52.74 = 55.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -13.8153362, upper bound: 13.8153362

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 29.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 27.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 57.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 57.23
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
IS_A2, status: Status.UNKNOWN, split count: 1, time: 57.23
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

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 40.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
time: 30.63 seconds

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

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 27.03 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8106329, upper bound: 13.8106329
time: 29.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 58.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 58.29
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 58.29
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 58.29
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 58.29
Output dim: 1, lower bound: -13.8106329, upper bound: 13.8106329

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.0065346, 25.9362659, -7.9497166, 25.9153900, -30.8227577, 30.7861633
1: -0.3328114, 26.8026581, -0.2966347, 26.7822990, -21.3829918, 21.3661957
2: -0.3728123, 25.8391533, -0.3644028, 25.8195705, -20.8778687, 20.8891907
3: -4.7156577, 22.5246754, -4.7297902, 22.5037251, -19.9566994, 19.9841576
4: -7.6287165, 22.5443230, -7.5941124, 22.5309391, -25.6491661, 25.6282845
5: -4.8135729, 24.8591385, -4.8248544, 24.8401585, -23.8104858, 23.8402252
6: -39.4240761, -4.1952667, -39.3976746, -4.2692089, -29.6741867, 29.7217484
7: -9.1832199, 23.4306870, -9.1607990, 23.4155388, -25.7839088, 25.7774582
8: -13.8365593, 20.0123749, -13.7656250, 19.9792824, -29.2483101, 29.2122879
9: -8.5709324, 22.5500813, -8.5160427, 22.5308075, -27.8490219, 27.8011322
10: -28.9714928, 17.9134922, -28.8985786, 17.9111443, -39.6904831, 39.6128235
11: -26.2373390, 6.8217578, -26.2216835, 6.8388143, -29.7008820, 29.6649818
12: -46.0929718, -8.3342552, -46.0875587, -8.3078709, -32.4054108, 32.3767853
13: -32.5820312, 13.2368746, -32.5976105, 13.2187595, -38.4913559, 38.5443420
14: -59.4513855, -2.1088943, -59.3805923, -2.0631914, -56.9347534, 56.8246307
15: -14.1962576, 18.8239822, -14.1460085, 18.7968407, -27.9967041, 27.9690132
16: -15.6517067, 22.2988453, -15.6203537, 22.2960873, -29.6665878, 29.6357918
17: -59.1999130, -6.9128685, -59.1571198, -6.8828144, -51.3984985, 51.3333893
18: -22.0452652, 16.3051605, -22.0186443, 16.3191719, -33.0574646, 33.0304565
19: -22.1620598, 6.0158892, -22.1337776, 6.0145607, -21.9731178, 21.9472809
20: -27.8413868, 0.5962219, -27.8093872, 0.5877681, -22.3634872, 22.3405685
21: -26.3115463, 7.3604474, -26.2736473, 7.3585339, -28.8568878, 28.8204002
22: -29.4831924, 5.2765150, -29.4514503, 5.2570257, -27.4764328, 27.4611855
23: -17.8319416, 12.2782249, -17.8134270, 12.2862473, -24.0361862, 24.0099983
24: -16.4944649, 13.6158543, -16.4640350, 13.6070232, -25.8169403, 25.7980804
25: -23.8337421, 8.8674994, -23.8061275, 8.8416567, -27.9185257, 27.9186287
26: -39.3347816, 4.0435910, -39.3113632, 4.0937042, -33.2956696, 33.2205963
27: -19.4981441, 14.8518963, -19.4709301, 14.8384085, -34.3365517, 34.3228264
28: -21.3847122, 11.4331913, -21.3622780, 11.4284840, -26.2838287, 26.2670059
29: -24.4019661, 6.3892808, -24.3778954, 6.3813791, -28.5993042, 28.5739822
30: -30.0154800, 5.8196831, -29.9856434, 5.8107719, -33.9295959, 33.9097824
31: -23.3566380, 7.6402512, -23.3287277, 7.6232810, -25.0240326, 25.0150146
32: -36.9956856, -2.6203108, -36.9797707, -2.6607132, -29.7719879, 29.7919083
33: -54.4402199, 0.0334740, -54.4280777, -0.0389481, -44.3990173, 44.4576263
34: -49.1584740, -7.0071802, -49.1260300, -7.0625324, -35.8234329, 35.8393631
35: -40.3096352, 3.8771925, -40.2729797, 3.8034725, -35.9585571, 35.9947281
36: -45.6192589, 1.3115759, -45.5700874, 1.2349319, -39.6593552, 39.6833572
37: -60.7243958, -9.8818970, -60.6760330, -9.9114084, -44.9030151, 44.8875809
38: -53.6681595, 4.0374212, -53.6058731, 3.9400206, -48.7571869, 48.8027649
39: -61.9401855, -4.5969629, -61.8984566, -4.6562538, -48.2602844, 48.2765350
40: -50.1814537, -9.1705923, -50.1619987, -9.1849918, -39.0897522, 39.0733414
41: -32.0027008, 7.2898536, -31.9825764, 7.2494235, -37.9440536, 37.9611816
42: -30.1104813, -0.3072248, -30.1077423, -0.3326883, -23.0857086, 23.1070862

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7527762, upper bound: 13.7776864
time: 29.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7744144, upper bound: 13.7782081
time: 35.56 seconds

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

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7533069, upper bound: 13.8093084
time: 27.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7749396, upper bound: 13.8098683
time: 30.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.1238070, 25.9976063, -7.9895363, 25.9162483, -30.9261055, 30.8918533
1: -0.4073372, 26.8775749, -0.3201246, 26.7833939, -21.4503860, 21.4656296
2: -0.4806204, 25.9478455, -0.4061098, 25.8216801, -20.9743996, 21.0393066
3: -4.8368936, 22.6429157, -4.7781167, 22.5043106, -20.0374069, 20.1053238
4: -7.7252474, 22.6408195, -7.6299610, 22.5334835, -25.7340965, 25.7616043
5: -4.9394836, 24.9690418, -4.8738899, 24.8415394, -23.9079437, 23.9893799
6: -39.4535561, -4.1745558, -39.4021912, -4.2687836, -29.7132721, 29.7470169
7: -9.2744884, 23.4808826, -9.1894274, 23.4167156, -25.8672867, 25.8528671
8: -13.9017353, 20.0979080, -13.7887831, 19.9828758, -29.3147926, 29.3233833
9: -8.5953903, 22.5718060, -8.5224037, 22.5335979, -27.9206848, 27.8106956
10: -29.0441017, 18.0003586, -28.9022026, 17.9361267, -39.8110962, 39.7042084
11: -26.4659061, 6.9345036, -26.2249336, 6.8857999, -29.9766693, 29.7458878
12: -46.2136841, -8.1775074, -46.0895081, -8.2468853, -32.5825500, 32.5035248
13: -32.6147537, 13.3272572, -32.6044464, 13.2261658, -38.5334778, 38.6376343
14: -59.6665192, -1.9525738, -59.3931694, -1.9981813, -57.2043152, 56.9421997
15: -14.2292433, 18.8719597, -14.1457176, 18.7985287, -28.0475693, 28.0084114
16: -15.7368088, 22.3120003, -15.6308117, 22.2895298, -29.7555923, 29.6589203
17: -59.4362602, -6.8020420, -59.1631851, -6.8377934, -51.6893616, 51.4152374
18: -22.2213097, 16.3870106, -22.0220318, 16.3535919, -33.2100754, 33.0769730
19: -22.3196106, 6.1041269, -22.1397743, 6.0514536, -22.1710892, 22.0172920
20: -27.9665394, 0.6897616, -27.8144722, 0.6259961, -22.5375748, 22.4148636
21: -26.5095673, 7.4803333, -26.2803192, 7.4082537, -29.1092606, 28.9217148
22: -29.5818291, 5.3282690, -29.4567242, 5.2750278, -27.5997009, 27.5157471
23: -17.9741135, 12.3670444, -17.8185921, 12.3212328, -24.2161942, 24.0810776
24: -16.5771370, 13.6762390, -16.4671593, 13.6318455, -25.9176102, 25.8431625
25: -23.9211082, 8.9340105, -23.8090210, 8.8678055, -28.0291367, 27.9694061
26: -39.5421791, 4.2098250, -39.3150902, 4.1647210, -33.5852661, 33.3511047
27: -19.6020317, 14.9188805, -19.4787998, 14.8675461, -34.4695778, 34.3976822
28: -21.5334759, 11.5337696, -21.3697929, 11.4696960, -26.4755707, 26.3493004
29: -24.5278111, 6.4420300, -24.3836575, 6.4011240, -28.7667999, 28.6306763
30: -30.1748810, 5.9312091, -29.9892826, 5.8526678, -34.1301270, 34.0037766
31: -23.5069141, 7.7162504, -23.3357201, 7.6548553, -25.2100143, 25.0722160
32: -37.0190315, -2.5858083, -36.9836121, -2.6537619, -29.8081818, 29.8304062
33: -54.5232086, 0.1740732, -54.4611626, -0.0304251, -44.4631042, 44.6466827
34: -49.1837616, -6.9668655, -49.1299286, -7.0586433, -35.8857193, 35.8705750
35: -40.3256874, 3.9142218, -40.2748032, 3.8048382, -35.9899292, 36.0537491
36: -45.6451950, 1.3436880, -45.5732231, 1.2399378, -39.7208633, 39.7206573
37: -60.7776642, -9.8437872, -60.6786423, -9.9027691, -44.9409332, 44.9545135
38: -53.7059288, 4.0758820, -53.6092033, 3.9431200, -48.7760468, 48.8966370
39: -61.9773102, -4.4926252, -61.9111023, -4.6462765, -48.3019867, 48.4013367
40: -50.2202835, -9.1271954, -50.1661186, -9.1816511, -39.1182251, 39.1423416
41: -32.0361366, 7.3186846, -31.9883423, 7.2523484, -37.9820709, 37.9956970
42: -30.1319847, -0.2742696, -30.1114750, -0.3297195, -23.1219978, 23.1408501

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7877091, upper bound: 13.7776864
time: 34.27 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8093456, upper bound: 13.7782081
time: 33.70 seconds

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

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7882391, upper bound: 13.8093084
time: 38.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8098681, upper bound: 13.8098683
time: 28.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 68.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7527762, upper bound: 13.7776864
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7744144, upper bound: 13.7782081
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7533069, upper bound: 13.8093084
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7749396, upper bound: 13.8098683
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7877091, upper bound: 13.7776864
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.8093456, upper bound: 13.7782081
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.7882391, upper bound: 13.8093084
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 68.73
Output dim: 1, lower bound: -13.8098681, upper bound: 13.8098683

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.8732729, 25.9125500, -7.8836069, 25.9133644, -30.6858749, 30.6953201
1: -0.2446203, 26.7834816, -0.2528553, 26.7814484, -21.2930908, 21.3022842
2: -0.3006213, 25.8175697, -0.3278913, 25.8184052, -20.8047333, 20.8314095
3: -4.6575212, 22.5101700, -4.6999884, 22.5014420, -19.8937912, 19.9363708
4: -7.5285044, 22.5289307, -7.5443659, 22.5294113, -25.5473671, 25.5606956
5: -4.7409177, 24.8360806, -4.7877326, 24.8373775, -23.7336311, 23.7780914
6: -39.4105911, -4.2705669, -39.3933258, -4.3060217, -29.6105881, 29.6385956
7: -9.0770741, 23.3998737, -9.1070042, 23.4137421, -25.6748009, 25.6912384
8: -13.7410851, 19.9892082, -13.7169304, 19.9761391, -29.1478386, 29.1394081
9: -8.4726095, 22.5186539, -8.4680300, 22.5292263, -27.7491913, 27.7206345
10: -28.8855686, 17.8801727, -28.8551559, 17.9060574, -39.5999527, 39.5379257
11: -26.1946030, 6.8011456, -26.2007637, 6.8293686, -29.6438599, 29.6216774
12: -46.0732727, -8.4219484, -46.0837860, -8.3508406, -32.3441544, 32.2732620
13: -32.5271149, 13.2129135, -32.5704269, 13.2101889, -38.4206924, 38.4749222
14: -59.3590736, -2.1399145, -59.3350830, -2.0672655, -56.8288422, 56.7327271
15: -14.1526031, 18.8041992, -14.1264362, 18.7860870, -27.9397812, 27.9241371
16: -15.5481701, 22.2577305, -15.5700989, 22.2947731, -29.5585632, 29.5427437
17: -59.1287270, -6.9384270, -59.1218719, -6.8921776, -51.3070831, 51.2440338
18: -22.0141373, 16.2661457, -22.0094700, 16.2999439, -33.0053864, 32.9791718
19: -22.1390038, 5.9790201, -22.1271076, 5.9957271, -21.9293861, 21.9019279
20: -27.8173523, 0.5485201, -27.8055000, 0.5639768, -22.3152046, 22.2886353
21: -26.2849617, 7.3246031, -26.2643452, 7.3405275, -28.8097992, 28.7723312
22: -29.4431076, 5.2175250, -29.4439278, 5.2267141, -27.4050522, 27.3927460
23: -17.8108158, 12.2346697, -17.8085918, 12.2650003, -23.9918976, 23.9595070
24: -16.4732933, 13.5796843, -16.4592323, 13.5888376, -25.7777023, 25.7568665
25: -23.8065033, 8.8113756, -23.8024139, 8.8130369, -27.8617706, 27.8576622
26: -39.2918091, 3.9629211, -39.3056526, 4.0530224, -33.2118530, 33.1345901
27: -19.4668884, 14.8095961, -19.4601593, 14.8174858, -34.2843742, 34.2697563
28: -21.3576565, 11.3750286, -21.3575859, 11.3986969, -26.2241211, 26.2008781
29: -24.3687363, 6.3432188, -24.3684502, 6.3577681, -28.5412598, 28.5169296
30: -30.0003433, 5.7959290, -29.9794083, 5.8000154, -33.8956146, 33.8733521
31: -23.3303413, 7.5939841, -23.3212509, 7.6001353, -24.9716263, 24.9588814
32: -36.9795990, -2.6763577, -36.9750175, -2.6884289, -29.7198410, 29.7273636
33: -54.3979950, -0.0458221, -54.4217224, -0.0785503, -44.3133545, 44.3699646
34: -49.1131363, -7.1023512, -49.1223373, -7.1096315, -35.7303391, 35.7399063
35: -40.2628784, 3.7881842, -40.2681961, 3.7587442, -35.8682785, 35.9012833
36: -45.5660400, 1.2090960, -45.5649872, 1.1837578, -39.5541153, 39.5718002
37: -60.6779404, -9.9565706, -60.6694260, -9.9494572, -44.8163147, 44.8037567
38: -53.5964394, 3.8998890, -53.5987930, 3.8722725, -48.6149292, 48.6511993
39: -61.8959885, -4.6608934, -61.8872452, -4.6878986, -48.1833649, 48.1989212
40: -50.1565208, -9.2079067, -50.1528625, -9.2029409, -39.0422668, 39.0231247
41: -31.9725590, 7.2225747, -31.9774208, 7.2160997, -37.8767548, 37.8873978
42: -30.0927887, -0.3708496, -30.1049690, -0.3642988, -23.0402718, 23.0378571

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7501891, upper bound: 13.7570591
time: 31.40 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7501891, upper bound: 13.7764058
time: 31.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.9994965, 25.9359169, -7.9466410, 25.9152527, -30.8129463, 30.7825737
1: -0.3283744, 26.8024254, -0.2947326, 26.7821903, -21.3644638, 21.3639717
2: -0.3689976, 25.8389645, -0.3627243, 25.8194561, -20.8569374, 20.8873024
3: -4.7122478, 22.5243492, -4.7281027, 22.5035534, -19.9366150, 19.9824982
4: -7.6233554, 22.5438938, -7.5917454, 22.5307617, -25.6233368, 25.6253853
5: -4.8099327, 24.8586082, -4.8231893, 24.8399372, -23.7868919, 23.8379631
6: -39.4235382, -4.1996508, -39.3974304, -4.2711639, -29.6772842, 29.7150726
7: -9.1780310, 23.4301682, -9.1584797, 23.4153099, -25.7475052, 25.7746124
8: -13.8316860, 20.0118065, -13.7634859, 19.9790344, -29.2240715, 29.2094994
9: -8.5661850, 22.5497799, -8.5139570, 22.5306549, -27.8240585, 27.7987022
10: -28.9671688, 17.9125633, -28.8966751, 17.9107323, -39.6611023, 39.6099854
11: -26.2337151, 6.8202343, -26.2201176, 6.8381276, -29.7007103, 29.6606178
12: -46.0922050, -8.3387499, -46.0872536, -8.3098488, -32.3999100, 32.3677559
13: -32.5786018, 13.2357140, -32.5960693, 13.2182455, -38.4837036, 38.5538254
14: -59.4457588, -2.1095648, -59.3776627, -2.0634308, -56.9243469, 56.8335114
15: -14.1936550, 18.8223629, -14.1448231, 18.7961426, -27.9925385, 27.9675140
16: -15.6467400, 22.2984314, -15.6181660, 22.2958984, -29.6184769, 29.6329575
17: -59.1944771, -6.9149771, -59.1544266, -6.8836880, -51.3837814, 51.3617249
18: -22.0438957, 16.3032074, -22.0179825, 16.3183384, -33.0549698, 33.0213165
19: -22.1612625, 6.0130987, -22.1334343, 6.0131555, -21.9713402, 21.9376221
20: -27.8407154, 0.5934839, -27.8090954, 0.5865655, -22.3615570, 22.3303375
21: -26.3102150, 7.3581891, -26.2730503, 7.3574548, -28.8538818, 28.8122940
22: -29.4817162, 5.2736101, -29.4507942, 5.2557392, -27.4735184, 27.4267731
23: -17.8310623, 12.2757349, -17.8130531, 12.2851830, -24.0341339, 23.9924583
24: -16.4932251, 13.6137352, -16.4635010, 13.6060333, -25.8148956, 25.7866669
25: -23.8331032, 8.8639965, -23.8058357, 8.8399076, -27.9164276, 27.8886948
26: -39.3337326, 4.0397077, -39.3109207, 4.0919142, -33.2928314, 33.1928024
27: -19.4963455, 14.8498249, -19.4701672, 14.8375111, -34.3338547, 34.3199921
28: -21.3838749, 11.4300661, -21.3619061, 11.4270792, -26.2814484, 26.2456970
29: -24.4000492, 6.3868618, -24.3770943, 6.3803310, -28.5961761, 28.5514259
30: -30.0125103, 5.8178935, -29.9843807, 5.8100085, -33.9240417, 33.9042587
31: -23.3558483, 7.6370292, -23.3283501, 7.6216588, -25.0220261, 25.0099792
32: -36.9946060, -2.6235032, -36.9793091, -2.6621304, -29.7733231, 29.7866669
33: -54.4391861, 0.0294857, -54.4275742, -0.0406933, -44.3964844, 44.4126053
34: -49.1576233, -7.0116372, -49.1256638, -7.0645084, -35.8203506, 35.8137360
35: -40.3087578, 3.8729553, -40.2725754, 3.8016510, -35.9558258, 35.9586563
36: -45.6183243, 1.3067389, -45.5697517, 1.2328491, -39.6561508, 39.6585693
37: -60.7231216, -9.8856068, -60.6754570, -9.9130287, -44.9003601, 44.8420181
38: -53.6674042, 4.0307856, -53.6054916, 3.9371548, -48.7530975, 48.7664490
39: -61.9382095, -4.6002665, -61.8976135, -4.6577072, -48.2570496, 48.2588272
40: -50.1800461, -9.1758394, -50.1613693, -9.1873560, -39.0871735, 39.0667648
41: -32.0018921, 7.2861400, -31.9822235, 7.2477679, -37.9454498, 37.9557571
42: -30.1099548, -0.3112302, -30.1075172, -0.3345013, -23.0831566, 23.0978928

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7718273, upper bound: 13.7575799
time: 34.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7718273, upper bound: 13.7769277
time: 28.51 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.9698296, 25.9156342, -8.0877094, 25.9384365, -30.8085175, 30.9000626
1: -0.3065033, 26.7855225, -0.3803110, 26.8054543, -21.3787994, 21.4169235
2: -0.3594394, 25.8197441, -0.4497080, 25.8430309, -20.8889771, 20.9330254
3: -4.7042327, 22.5164757, -4.7961221, 22.5296021, -19.9700279, 20.0231133
4: -7.6019039, 22.5320625, -7.6971970, 22.5494308, -25.6382828, 25.6900711
5: -4.7909784, 24.8394356, -4.8912649, 24.8616409, -23.8092842, 23.8657188
6: -39.4162369, -4.1985283, -39.4315453, -4.1527786, -29.7674179, 29.7489853
7: -9.1319475, 23.4017467, -9.2200317, 23.4326305, -25.7515106, 25.7858582
8: -13.8343611, 19.9939232, -13.9097300, 20.0189056, -29.2832108, 29.2996826
9: -8.5306482, 22.5232449, -8.5897207, 22.5579166, -27.8377075, 27.8320389
10: -28.9562893, 17.8894882, -29.0011253, 17.9566193, -39.7288589, 39.6686707
11: -26.2030907, 6.8398671, -26.2290688, 6.9107237, -29.7293396, 29.6901016
12: -46.0796738, -8.3747520, -46.0979919, -8.2488956, -32.4662476, 32.3386154
13: -32.5408897, 13.2319841, -32.6029739, 13.2562637, -38.5715942, 38.5334396
14: -59.4357605, -2.1316881, -59.4971504, -2.0171566, -56.9596863, 56.9033356
15: -14.2222309, 18.8121662, -14.2744865, 18.8237495, -28.0462189, 28.0753403
16: -15.5892925, 22.2595196, -15.6547661, 22.3151970, -29.6256256, 29.5872955
17: -59.1739349, -6.9285078, -59.2165680, -6.8529758, -51.4185028, 51.3474121
18: -22.0289783, 16.2966118, -22.0560036, 16.3631325, -33.0972595, 33.0939102
19: -22.1484833, 6.0255494, -22.1730537, 6.0914569, -22.0240288, 21.9905510
20: -27.8236237, 0.6029391, -27.8506775, 0.6763415, -22.4169731, 22.3869438
21: -26.2975979, 7.3870716, -26.3239384, 7.4696183, -28.9377670, 28.8929596
22: -29.4499722, 5.2606320, -29.4897118, 5.3137732, -27.4736557, 27.4872284
23: -17.8176880, 12.2694454, -17.8409252, 12.3376045, -24.0568008, 24.0254288
24: -16.4796638, 13.6184883, -16.5004692, 13.6684275, -25.8539963, 25.8361282
25: -23.8117065, 8.8661251, -23.8393021, 8.9278851, -27.9584045, 27.9592781
26: -39.2993736, 4.0066309, -39.3417015, 4.1407189, -33.3012848, 33.2194901
27: -19.4803047, 14.8551006, -19.5115604, 14.9118862, -34.3921890, 34.3666611
28: -21.3642349, 11.4296818, -21.3967323, 11.5117426, -26.3323135, 26.2942810
29: -24.3757591, 6.3748593, -24.4073181, 6.4241543, -28.5922394, 28.5918121
30: -30.0064487, 5.8533235, -30.0201912, 5.9209085, -34.0172119, 33.9722900
31: -23.3401718, 7.6473203, -23.3683243, 7.7110147, -25.0851059, 25.0559845
32: -36.9860687, -2.6279106, -37.0028763, -2.5860090, -29.8175125, 29.8072815
33: -54.4081039, 0.0307093, -54.4890442, 0.0783577, -44.4271851, 44.5180435
34: -49.1219406, -7.0469131, -49.1725693, -6.9963775, -35.8399124, 35.8461990
35: -40.2721977, 3.8576870, -40.3279991, 3.9003754, -35.9865723, 36.0336304
36: -45.5726776, 1.2871971, -45.6251717, 1.3434887, -39.6761780, 39.7102890
37: -60.6890678, -9.9070034, -60.7328262, -9.8478422, -44.8854218, 44.9180069
38: -53.6077614, 3.9974785, -53.6770325, 4.0731707, -48.7595367, 48.8232346
39: -61.9066620, -4.5944252, -61.9570694, -4.5516844, -48.2809448, 48.3395844
40: -50.1697655, -9.1924820, -50.1915054, -9.1697884, -39.0877533, 39.0962906
41: -31.9807587, 7.2687240, -32.0138626, 7.3129835, -37.9810104, 37.9727097
42: -30.0980492, -0.3378677, -30.1177273, -0.2936296, -23.1224632, 23.0925026

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7507197, upper bound: 13.7886817
time: 37.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7501891, upper bound: 13.8080303
time: 34.29 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.0960674, 25.9389954, -8.1507998, 25.9403191, -30.9356384, 30.9873352
1: -0.3902359, 26.8044624, -0.4222212, 26.8062057, -21.4501648, 21.4786415
2: -0.4278345, 25.8411160, -0.4845257, 25.8441010, -20.9412117, 20.9889450
3: -4.7590027, 22.5306664, -4.8242316, 22.5317345, -20.0128937, 20.0692329
4: -7.6967831, 22.5470333, -7.7446327, 22.5508118, -25.7142792, 25.7547913
5: -4.8599424, 24.8619175, -4.9267893, 24.8641872, -23.8625984, 23.9256439
6: -39.4291458, -4.1276503, -39.4356155, -4.1178780, -29.8341370, 29.8255386
7: -9.2328730, 23.4320641, -9.2716017, 23.4342098, -25.8242340, 25.8692474
8: -13.9249964, 20.0165138, -13.9562597, 20.0218201, -29.3595238, 29.3698158
9: -8.6242781, 22.5543728, -8.6357269, 22.5593586, -27.9125748, 27.9101448
10: -29.0378990, 17.9218235, -29.0426674, 17.9613171, -39.7900696, 39.7407684
11: -26.2421703, 6.8590312, -26.2483845, 6.9195471, -29.7862625, 29.7290573
12: -46.0986176, -8.2915440, -46.1014519, -8.2078533, -32.5221024, 32.4332161
13: -32.5923576, 13.2547560, -32.6285477, 13.2643623, -38.6345520, 38.6124191
14: -59.5225258, -2.1013460, -59.5396233, -2.0134277, -57.0551758, 57.0041351
15: -14.2634182, 18.8303471, -14.2930193, 18.8337784, -28.0990067, 28.1187820
16: -15.6878529, 22.3001747, -15.7029114, 22.3163357, -29.6855011, 29.6775284
17: -59.2396927, -6.9050360, -59.2491913, -6.8445015, -51.4951172, 51.4650879
18: -22.0587254, 16.3337040, -22.0645199, 16.3815746, -33.1468201, 33.1360550
19: -22.1707268, 6.0596409, -22.1793842, 6.1088858, -22.0659981, 22.0262299
20: -27.8470020, 0.6479206, -27.8542709, 0.6989422, -22.4632874, 22.4286308
21: -26.3228302, 7.4206834, -26.3326626, 7.4865594, -28.9818497, 28.9329147
22: -29.4885712, 5.3167086, -29.4965458, 5.3428383, -27.5421143, 27.5212936
23: -17.8379135, 12.3104935, -17.8452988, 12.3578377, -24.0990372, 24.0583992
24: -16.4996262, 13.6525507, -16.5047340, 13.6855650, -25.8911896, 25.8659668
25: -23.8383274, 8.9187069, -23.8427162, 8.9548111, -28.0129852, 27.9903259
26: -39.3412781, 4.0833960, -39.3469620, 4.1795535, -33.3822632, 33.2777481
27: -19.5097389, 14.8953295, -19.5215187, 14.9319477, -34.4416885, 34.4168472
28: -21.3904819, 11.4847078, -21.4010811, 11.5401306, -26.3896637, 26.3390770
29: -24.4070702, 6.4185257, -24.4158936, 6.4466920, -28.6471710, 28.6262970
30: -30.0185814, 5.8753104, -30.0251350, 5.9309011, -34.0456009, 34.0032043
31: -23.3656616, 7.6903930, -23.3753681, 7.7325506, -25.1355286, 25.1071091
32: -37.0010910, -2.5750241, -37.0071411, -2.5596428, -29.8709717, 29.8665848
33: -54.4493713, 0.1060019, -54.4949188, 0.1161385, -44.5104065, 44.5606689
34: -49.1664238, -6.9562101, -49.1759148, -6.9512529, -35.9299545, 35.9200211
35: -40.3179703, 3.9424725, -40.3323212, 3.9433079, -36.0740814, 36.0909729
36: -45.6250229, 1.3848248, -45.6298943, 1.3926086, -39.7782516, 39.7970657
37: -60.7342148, -9.8360977, -60.7388649, -9.8113976, -44.9694519, 44.9562759
38: -53.6786652, 4.1283245, -53.6836853, 4.1380272, -48.8977051, 48.9384460
39: -61.9488945, -4.5337257, -61.9673271, -4.5215111, -48.3545990, 48.3994446
40: -50.1932983, -9.1603670, -50.1999741, -9.1542168, -39.1327057, 39.1400375
41: -32.0100327, 7.3322697, -32.0187225, 7.3446679, -38.0496826, 38.0411072
42: -30.1152039, -0.2782249, -30.1202888, -0.2638340, -23.1653709, 23.1526070

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7723525, upper bound: 13.7892414
time: 37.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
time: 30.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.9904699, 25.9738731, -7.9234047, 25.9142036, -30.7891808, 30.8010216
1: -0.3191867, 26.8583851, -0.2763343, 26.7825356, -21.3604889, 21.4017258
2: -0.4084191, 25.9262371, -0.3696384, 25.8204956, -20.9012604, 20.9815483
3: -4.7787628, 22.6284237, -4.7482648, 22.5020561, -19.9744682, 20.0575600
4: -7.6250296, 22.6254368, -7.5802317, 22.5319099, -25.6323547, 25.6940193
5: -4.8668280, 24.9459896, -4.8368068, 24.8387146, -23.8310242, 23.9272003
6: -39.4401016, -4.2498446, -39.3978004, -4.3055840, -29.6497192, 29.6638336
7: -9.1683426, 23.4500599, -9.1356354, 23.4149227, -25.7581711, 25.7666626
8: -13.8062201, 20.0747490, -13.7400913, 19.9796925, -29.2142868, 29.2505417
9: -8.4970827, 22.5403557, -8.4744072, 22.5319901, -27.8208923, 27.7302094
10: -28.9582176, 17.9669914, -28.8588200, 17.9310131, -39.7205658, 39.6292114
11: -26.4231567, 6.9138737, -26.2040558, 6.8763781, -29.9196472, 29.7025986
12: -46.1939850, -8.2651625, -46.0856476, -8.2898865, -32.5212555, 32.3999481
13: -32.5598183, 13.3032770, -32.5772667, 13.2176733, -38.4628983, 38.5681229
14: -59.5742264, -1.9835882, -59.3476639, -2.0022411, -57.0983734, 56.8502197
15: -14.1855812, 18.8522072, -14.1261768, 18.7877865, -27.9906998, 27.9635277
16: -15.6333294, 22.2709122, -15.5805931, 22.2882118, -29.6475754, 29.5658989
17: -59.3650475, -6.8275862, -59.1280060, -6.8471575, -51.5980377, 51.3258514
18: -22.1901169, 16.3480148, -22.0129013, 16.3343391, -33.1579590, 33.0256729
19: -22.2965355, 6.0672646, -22.1331215, 6.0326147, -22.1273575, 21.9719429
20: -27.9424801, 0.6420326, -27.8105888, 0.6022077, -22.4893608, 22.3629379
21: -26.4829865, 7.4444594, -26.2709732, 7.3902502, -29.0622025, 28.8736496
22: -29.5416374, 5.2693458, -29.4491997, 5.2447877, -27.5283279, 27.4472733
23: -17.9529762, 12.3235092, -17.8137550, 12.2999897, -24.1719055, 24.0305672
24: -16.5559654, 13.6400404, -16.4623566, 13.6136923, -25.8782883, 25.8018875
25: -23.8938446, 8.8779726, -23.8053188, 8.8391953, -27.9723816, 27.9083786
26: -39.4992485, 4.1290874, -39.3093452, 4.1241274, -33.5014648, 33.2649918
27: -19.5707550, 14.8765812, -19.4680405, 14.8466167, -34.4173737, 34.3446198
28: -21.5064812, 11.4755974, -21.3651161, 11.4399681, -26.4158936, 26.2832031
29: -24.4946060, 6.3960066, -24.3741684, 6.3774943, -28.7088394, 28.5736198
30: -30.1597118, 5.9075303, -29.9829559, 5.8419275, -34.0961227, 33.9673462
31: -23.4806499, 7.6699653, -23.3282547, 7.6317186, -25.1576157, 25.0160828
32: -37.0029335, -2.6418629, -36.9788857, -2.6815171, -29.7560883, 29.7658081
33: -54.4809723, 0.0948257, -54.4548225, -0.0699863, -44.3774567, 44.5590668
34: -49.1384048, -7.0620422, -49.1263046, -7.1057048, -35.7925873, 35.7710876
35: -40.2788925, 3.8252144, -40.2700310, 3.7601337, -35.8996887, 35.9603195
36: -45.5919876, 1.2411823, -45.5680695, 1.1887569, -39.6156693, 39.6090775
37: -60.7311592, -9.9184942, -60.6720276, -9.9408226, -44.8541870, 44.8707047
38: -53.6342049, 3.9383678, -53.6021042, 3.8753719, -48.6337433, 48.7450943
39: -61.9331284, -4.5566874, -61.8998871, -4.6778965, -48.2249146, 48.3238144
40: -50.1953125, -9.1645861, -50.1569901, -9.1996355, -39.0707397, 39.0921173
41: -32.0059967, 7.2514648, -31.9831734, 7.2190228, -37.9147949, 37.9218674
42: -30.1142979, -0.3379059, -30.1086750, -0.3612785, -23.0765533, 23.0716209

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7501890, upper bound: 13.7570591
time: 24.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7864282, upper bound: 13.7764058
time: 28.44 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1167498, 25.9972343, -7.9864635, 25.9161072, -30.9163055, 30.8882446
1: -0.4028878, 26.8773537, -0.3182135, 26.7832851, -21.4318542, 21.4634247
2: -0.4768496, 25.9476318, -0.4044218, 25.8215771, -20.9534760, 21.0374451
3: -4.8334899, 22.6426754, -4.7764015, 22.5041847, -20.0173264, 20.1036491
4: -7.7198782, 22.6403885, -7.6276007, 22.5332890, -25.7082863, 25.7586594
5: -4.9358044, 24.9685020, -4.8722930, 24.8413010, -23.8843689, 23.9871368
6: -39.4530220, -4.1790047, -39.4019279, -4.2707672, -29.7163773, 29.7403336
7: -9.2692404, 23.4803429, -9.1871767, 23.4165154, -25.8308907, 25.8499985
8: -13.8968430, 20.0973587, -13.7866707, 19.9826241, -29.2905884, 29.3206139
9: -8.5906811, 22.5715046, -8.5203609, 22.5334492, -27.8957443, 27.8082504
10: -29.0397816, 17.9994011, -28.9003754, 17.9357452, -39.7817764, 39.7012634
11: -26.4622746, 6.9330106, -26.2233543, 6.8851638, -29.9765472, 29.7415237
12: -46.2129517, -8.1819696, -46.0891342, -8.2489033, -32.5770416, 32.4944649
13: -32.6112709, 13.3260670, -32.6028175, 13.2256641, -38.5258179, 38.6471176
14: -59.6607895, -1.9531078, -59.3901329, -1.9984350, -57.1939545, 56.9509888
15: -14.2266273, 18.8703423, -14.1445503, 18.7978268, -28.0434113, 28.0069046
16: -15.7318039, 22.3115864, -15.6286774, 22.2893562, -29.7074814, 29.6560783
17: -59.4308434, -6.8041439, -59.1605492, -6.8386965, -51.6746674, 51.4435425
18: -22.2199249, 16.3850403, -22.0214329, 16.3527317, -33.2075806, 33.0678406
19: -22.3187904, 6.1013188, -22.1394463, 6.0500722, -22.1693039, 22.0076294
20: -27.9658680, 0.6870079, -27.8142052, 0.6247783, -22.5356407, 22.4046249
21: -26.5081902, 7.4780521, -26.2797203, 7.4072232, -29.1062775, 28.9135971
22: -29.5803108, 5.3253932, -29.4560242, 5.2737741, -27.5967636, 27.4813232
23: -17.9732075, 12.3645668, -17.8181820, 12.3201618, -24.2141190, 24.0635147
24: -16.5759277, 13.6741428, -16.4665909, 13.6308565, -25.9155502, 25.8317490
25: -23.9204407, 8.9305363, -23.8087502, 8.8660774, -28.0270233, 27.9394379
26: -39.5411148, 4.2059250, -39.3145790, 4.1629353, -33.5824661, 33.3233109
27: -19.6002502, 14.9167938, -19.4779758, 14.8666515, -34.4669037, 34.3947678
28: -21.5326767, 11.5306101, -21.3694420, 11.4683399, -26.4731903, 26.3279991
29: -24.5259209, 6.4396477, -24.3828354, 6.4000793, -28.7636719, 28.6080933
30: -30.1718464, 5.9294662, -29.9879322, 5.8519053, -34.1245575, 33.9982376
31: -23.5061302, 7.7129660, -23.3353500, 7.6532669, -25.2080231, 25.0671730
32: -37.0179634, -2.5890183, -36.9831734, -2.6552248, -29.8095169, 29.8251266
33: -54.5221519, 0.1700745, -54.4606934, -0.0321522, -44.4606628, 44.6016846
34: -49.1829262, -6.9713650, -49.1296158, -7.0605564, -35.8825912, 35.8448868
35: -40.3247337, 3.9099884, -40.2743378, 3.8029995, -35.9872360, 36.0176697
36: -45.6443214, 1.3388233, -45.5727921, 1.2378540, -39.7176819, 39.6958771
37: -60.7764206, -9.8475094, -60.6781044, -9.9043579, -44.9382935, 44.9089661
38: -53.7051430, 4.0692062, -53.6088524, 3.9402275, -48.7718964, 48.8603134
39: -61.9753990, -4.4960136, -61.9102516, -4.6477261, -48.2987213, 48.3836136
40: -50.2188568, -9.1324968, -50.1654854, -9.1840734, -39.1156464, 39.1357803
41: -32.0353203, 7.3149514, -31.9879684, 7.2506642, -37.9834595, 37.9902344
42: -30.1314964, -0.2782784, -30.1112576, -0.3315067, -23.1194344, 23.1316872

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8080650, upper bound: 13.7575799
time: 41.25 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8080650, upper bound: 13.7769277
time: 31.06 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0870705, 25.9769459, -8.1274967, 25.9392624, -30.9118423, 31.0057449
1: -0.3809915, 26.8604412, -0.4038348, 26.8065586, -21.4461861, 21.5163651
2: -0.4672842, 25.9284115, -0.4914417, 25.8451195, -20.9855194, 21.0831642
3: -4.8254924, 22.6347332, -4.8444209, 22.5302086, -20.0507469, 20.1442757
4: -7.6984625, 22.6285362, -7.7330432, 22.5519638, -25.7232323, 25.8233795
5: -4.9168620, 24.9493465, -4.9403067, 24.8630085, -23.9067268, 24.0148621
6: -39.4457245, -4.1778173, -39.4360275, -4.1523218, -29.8065491, 29.7742233
7: -9.2231989, 23.4519691, -9.2487621, 23.4338207, -25.8349190, 25.8612556
8: -13.8995275, 20.0794678, -13.9328718, 20.0224876, -29.3496704, 29.4107742
9: -8.5551109, 22.5449600, -8.5961285, 22.5606575, -27.9093857, 27.8416405
10: -29.0289364, 17.9763298, -29.0048103, 17.9816475, -39.8495178, 39.7600098
11: -26.4316025, 6.9526196, -26.2323532, 6.9577689, -30.0050888, 29.7709885
12: -46.2003937, -8.2179413, -46.0999222, -8.1879377, -32.6434479, 32.4653778
13: -32.5735588, 13.3223495, -32.6098061, 13.2637720, -38.6137543, 38.6267548
14: -59.6509399, -1.9752674, -59.5096321, -1.9521933, -57.2291870, 57.0207825
15: -14.2552433, 18.8601761, -14.2742386, 18.8254261, -28.0971375, 28.1147537
16: -15.6744118, 22.2726364, -15.6652851, 22.3086357, -29.7145996, 29.6104507
17: -59.4102898, -6.8176565, -59.2226639, -6.8079758, -51.7092667, 51.4292297
18: -22.2050171, 16.3785000, -22.0593796, 16.3975258, -33.2499161, 33.1404190
19: -22.3060074, 6.1137829, -22.1791000, 6.1283393, -22.2220230, 22.0605583
20: -27.9487629, 0.6964679, -27.8557892, 0.7146239, -22.5910950, 22.4612274
21: -26.4955921, 7.5069723, -26.3305359, 7.5193696, -29.1901398, 28.9943008
22: -29.5485611, 5.3123565, -29.4949150, 5.3318272, -27.5969543, 27.5417671
23: -17.9598083, 12.3582373, -17.8460350, 12.3726006, -24.2368164, 24.0964813
24: -16.5623322, 13.6788616, -16.5035934, 13.6932573, -25.9546127, 25.8811798
25: -23.8990479, 8.9327297, -23.8422108, 8.9540987, -28.0689774, 28.0100632
26: -39.5067673, 4.1728625, -39.3454285, 4.2117333, -33.5909119, 33.3499908
27: -19.5841713, 14.9221001, -19.5194283, 14.9410639, -34.5252342, 34.4415283
28: -21.5130806, 11.5303135, -21.4042797, 11.5529633, -26.5240479, 26.3765717
29: -24.5016174, 6.4276342, -24.4130745, 6.4439015, -28.7598114, 28.6484680
30: -30.1657944, 5.9649143, -30.0237961, 5.9628720, -34.2176971, 34.0662918
31: -23.4904957, 7.7233090, -23.3752728, 7.7425475, -25.2710876, 25.1131668
32: -37.0094337, -2.5933528, -37.0067101, -2.5790596, -29.8537140, 29.8457870
33: -54.4911385, 0.1712761, -54.5221596, 0.0868225, -44.4913635, 44.7071304
34: -49.1472816, -7.0066700, -49.1765747, -6.9924479, -35.9021378, 35.8774033
35: -40.2882156, 3.8946896, -40.3297577, 3.9017363, -36.0178986, 36.0926819
36: -45.5986748, 1.3193216, -45.6282349, 1.3484545, -39.7376556, 39.7476425
37: -60.7422943, -9.8689785, -60.7354355, -9.8392172, -44.9232330, 44.9849319
38: -53.6455307, 4.0359325, -53.6803436, 4.0763264, -48.7783966, 48.9171219
39: -61.9437943, -4.4901400, -61.9696808, -4.5417471, -48.3225861, 48.4643707
40: -50.2085724, -9.1491518, -50.1955986, -9.1664896, -39.1162720, 39.1653137
41: -32.0141945, 7.2975655, -32.0195961, 7.3159633, -38.0189972, 38.0071564
42: -30.1195850, -0.3048930, -30.1214027, -0.2906909, -23.1587791, 23.1263237

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
time: 32.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
time: 26.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.2133408, 26.0003242, -8.1905737, 25.9411583, -31.0389824, 31.0930176
1: -0.4647560, 26.8793945, -0.4456878, 26.8072853, -21.5175476, 21.5780525
2: -0.5356827, 25.9497852, -0.5262647, 25.8461838, -21.0377426, 21.1390610
3: -4.8802934, 22.6489716, -4.8725290, 22.5323048, -20.0936127, 20.1903992
4: -7.7933154, 22.6435032, -7.7804413, 22.5533066, -25.7992325, 25.8880959
5: -4.9858441, 24.9718781, -4.9758339, 24.8655548, -23.9600410, 24.0748138
6: -39.4586143, -4.1069717, -39.4400940, -4.1174850, -29.8732605, 29.8507309
7: -9.3241253, 23.4822502, -9.3002586, 23.4353790, -25.9076271, 25.9446411
8: -13.9901485, 20.1020508, -13.9794531, 20.0254154, -29.4259949, 29.4809036
9: -8.6487188, 22.5760784, -8.6420631, 22.5621376, -27.9842072, 27.9196968
10: -29.1104889, 18.0087318, -29.0463295, 17.9863281, -39.9106445, 39.8320770
11: -26.4707108, 6.9717922, -26.2516422, 6.9665504, -30.0620346, 29.8099022
12: -46.2193375, -8.1347103, -46.1034050, -8.1468811, -32.6992188, 32.5600052
13: -32.6250725, 13.3451090, -32.6354294, 13.2717819, -38.6767502, 38.7056808
14: -59.7375603, -1.9449186, -59.5521469, -1.9484081, -57.3246918, 57.1216583
15: -14.2963772, 18.8783054, -14.2927809, 18.8354702, -28.1498871, 28.1581535
16: -15.7729321, 22.3133259, -15.7133751, 22.3097687, -29.7745132, 29.7006836
17: -59.4761200, -6.7941494, -59.2552299, -6.7994833, -51.7860413, 51.5469055
18: -22.2347527, 16.4155712, -22.0679436, 16.4159813, -33.2995453, 33.1825943
19: -22.3282604, 6.1478577, -22.1854095, 6.1458340, -22.2640076, 22.0962524
20: -27.9721298, 0.7414794, -27.8593578, 0.7371235, -22.6373672, 22.5029297
21: -26.5207634, 7.5405216, -26.3393688, 7.5363026, -29.2341995, 29.0342484
22: -29.5871353, 5.3684564, -29.5017738, 5.3608069, -27.6653671, 27.5758171
23: -17.9800568, 12.3992977, -17.8504791, 12.3928070, -24.2789993, 24.1294746
24: -16.5822964, 13.7129183, -16.5078430, 13.7104292, -25.9918594, 25.9110794
25: -23.9256401, 8.9853354, -23.8456097, 8.9810257, -28.1236038, 28.0411415
26: -39.5486641, 4.2496328, -39.3506737, 4.2506289, -33.6719055, 33.4082565
27: -19.6135941, 14.9623938, -19.5293674, 14.9611034, -34.5746994, 34.4917603
28: -21.5392303, 11.5853167, -21.4085960, 11.5813284, -26.5813904, 26.4213562
29: -24.5329113, 6.4713049, -24.4216461, 6.4664764, -28.8146591, 28.6829529
30: -30.1779556, 5.9868798, -30.0287323, 5.9728765, -34.2461090, 34.0972290
31: -23.5159550, 7.7663403, -23.3823338, 7.7641344, -25.3214722, 25.1642914
32: -37.0244980, -2.5404878, -37.0109406, -2.5527129, -29.9071503, 29.9051361
33: -54.5323906, 0.2465496, -54.5280533, 0.1246738, -44.5745239, 44.7497025
34: -49.1918259, -6.9159594, -49.1798477, -6.9473319, -35.9921875, 35.9511566
35: -40.3340340, 3.9795046, -40.3341866, 3.9446630, -36.1054688, 36.1500168
36: -45.6508980, 1.4169312, -45.6329727, 1.3975973, -39.8397141, 39.8343887
37: -60.7874832, -9.7979965, -60.7414932, -9.8027840, -45.0073853, 45.0231552
38: -53.7164001, 4.1668053, -53.6870117, 4.1411629, -48.9165497, 49.0323563
39: -61.9860306, -4.4295349, -61.9799881, -4.5115061, -48.3962555, 48.5241547
40: -50.2321091, -9.1170359, -50.2040863, -9.1509123, -39.1611481, 39.2089844
41: -32.0435181, 7.3611422, -32.0244370, 7.3476152, -38.0877075, 38.0755615
42: -30.1367455, -0.2452636, -30.1240082, -0.2608380, -23.2016678, 23.1864128

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085879, upper bound: 13.7892414
time: 31.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880
time: 33.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 66.85 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7501891, upper bound: 13.7570591
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7501891, upper bound: 13.7764058
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7718273, upper bound: 13.7575799
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7718273, upper bound: 13.7769277
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7507197, upper bound: 13.7886817
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7501891, upper bound: 13.8080303
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7723525, upper bound: 13.7892414
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7501890, upper bound: 13.7570591
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7864282, upper bound: 13.7764058
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.8080650, upper bound: 13.7575799
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.8080650, upper bound: 13.7769277
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.8085879, upper bound: 13.7892414
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.85
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.8146734, 25.9115448, -7.7577600, 25.8931866, -30.6049576, 30.5679932
1: -0.2115784, 26.7822571, -0.1802979, 26.7562141, -21.2307625, 21.2241287
2: -0.2394145, 25.8155136, -0.2018273, 25.7776775, -20.7029495, 20.7060776
3: -4.5888209, 22.5078506, -4.5597277, 22.4505253, -19.7675323, 19.7887573
4: -7.4723635, 22.5258560, -7.4267664, 22.4863167, -25.4505844, 25.4419746
5: -4.6653538, 24.8319359, -4.6325288, 24.7864838, -23.6044998, 23.6166153
6: -39.4039841, -4.2815113, -39.3728867, -4.3340406, -29.5731354, 29.6027374
7: -9.0296631, 23.3974113, -9.0001974, 23.3979454, -25.5925140, 25.5764160
8: -13.6935120, 19.9849396, -13.6168003, 19.9312134, -29.0534019, 29.0346603
9: -8.4627485, 22.4956970, -8.4395523, 22.4777622, -27.6852493, 27.6473465
10: -28.8793850, 17.8244095, -28.8047714, 17.7858963, -39.4751663, 39.4343033
11: -26.1907158, 6.7543354, -26.1204357, 6.7337980, -29.5338898, 29.4833298
12: -46.0699997, -8.5406761, -46.0079880, -8.5941658, -32.1001053, 32.0802155
13: -32.5155411, 13.1971493, -32.5438194, 13.1683102, -38.3666840, 38.4119949
14: -59.3391495, -2.2396297, -59.2188568, -2.2698784, -56.6062012, 56.5178680
15: -14.1226902, 18.8015881, -14.0586653, 18.7633209, -27.8829498, 27.8504791
16: -15.5331841, 22.2447357, -15.5204277, 22.2647095, -29.5123444, 29.4776001
17: -59.1211014, -6.9939985, -59.0308762, -7.0081997, -51.1872101, 51.1000671
18: -22.0084667, 16.2314434, -21.9426556, 16.2324638, -32.9441833, 32.9217529
19: -22.1292629, 5.9431725, -22.0646992, 5.9236603, -21.8465843, 21.7979202
20: -27.8093472, 0.4996386, -27.7484856, 0.4644918, -22.1978722, 22.1708145
21: -26.2758484, 7.2671976, -26.1855965, 7.2242227, -28.6758804, 28.6246223
22: -29.4361191, 5.1973562, -29.4178886, 5.1782684, -27.3422012, 27.3001671
23: -17.8026962, 12.2004356, -17.7519798, 12.1943874, -23.9149933, 23.8809204
24: -16.4667530, 13.5689688, -16.4399071, 13.5657339, -25.7408142, 25.7313538
25: -23.8021317, 8.7861204, -23.7739868, 8.7599106, -27.8088531, 27.8123055
26: -39.2865028, 3.8656321, -39.2184525, 3.8596611, -33.0097656, 32.9504013
27: -19.4531174, 14.8006468, -19.4245033, 14.7956476, -34.2487640, 34.2251511
28: -21.3461132, 11.3364849, -21.2999382, 11.3198586, -26.1343384, 26.1077118
29: -24.3615341, 6.3207746, -24.3283653, 6.3079772, -28.4772034, 28.4226761
30: -29.9964771, 5.7627749, -29.9381180, 5.7286935, -33.8208008, 33.8045578
31: -23.3185158, 7.5677762, -23.2592793, 7.5466332, -24.9121246, 24.8816605
32: -36.9746628, -2.7080908, -36.9522476, -2.7563257, -29.6475754, 29.6672897
33: -54.3536453, -0.0578070, -54.3298264, -0.1453619, -44.2046967, 44.2675781
34: -49.0934372, -7.1084862, -49.0780716, -7.1467185, -35.6793900, 35.6883087
35: -40.2463799, 3.7862215, -40.2323074, 3.7393475, -35.8245392, 35.8608246
36: -45.5610428, 1.1943378, -45.5492172, 1.1442509, -39.5001678, 39.5177460
37: -60.6706047, -9.9829445, -60.6338005, -10.0069618, -44.7378235, 44.7467270
38: -53.5898628, 3.8801699, -53.5679550, 3.8231106, -48.5474701, 48.6088333
39: -61.8863144, -4.6761904, -61.8632889, -4.7361422, -48.1153412, 48.1503906
40: -50.1463432, -9.2134447, -50.1191177, -9.2244816, -38.9918671, 38.9858475
41: -31.9632664, 7.2115064, -31.9497681, 7.1898055, -37.8381500, 37.8463974
42: -30.0875206, -0.3882990, -30.0895252, -0.4055128, -22.9921722, 22.9968300

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7351426
time: 28.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7561999
time: 30.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.8707962, 25.9124508, -7.8783445, 25.9131565, -30.6834717, 30.6674194
1: -0.2425046, 26.7833767, -0.2480960, 26.7812386, -21.2911835, 21.2908936
2: -0.2979999, 25.8174553, -0.3221385, 25.8181915, -20.8021126, 20.7961426
3: -4.6546760, 22.5099583, -4.6936255, 22.5010185, -19.8915482, 19.8618622
4: -7.5260572, 22.5287189, -7.5388498, 22.5289879, -25.5441742, 25.5118179
5: -4.7377939, 24.8357906, -4.7806535, 24.8367519, -23.7309303, 23.7071991
6: -39.4103355, -4.2729521, -39.3927307, -4.3113737, -29.6007767, 29.6356964
7: -9.0731773, 23.3997650, -9.0990267, 23.4135056, -25.6757431, 25.6841545
8: -13.7387409, 19.9889526, -13.7117453, 19.9755592, -29.1452560, 29.1266747
9: -8.4721212, 22.5168037, -8.4669724, 22.5250645, -27.7406845, 27.7333221
10: -28.8853474, 17.8772907, -28.8546524, 17.8995495, -39.5795364, 39.5344086
11: -26.1943207, 6.7977905, -26.2001686, 6.8218989, -29.5315247, 29.6181755
12: -46.0730934, -8.4270401, -46.0833397, -8.3621941, -32.2734833, 32.2679443
13: -32.5238914, 13.2121201, -32.5630569, 13.2085590, -38.4052887, 38.4971848
14: -59.3580666, -2.1440144, -59.3329735, -2.0763702, -56.7403107, 56.7264557
15: -14.1452789, 18.8039608, -14.1099052, 18.7855759, -27.9303055, 27.8793106
16: -15.5473156, 22.2540112, -15.5681095, 22.2863693, -29.5376663, 29.5367203
17: -59.1282654, -6.9399204, -59.1208992, -6.8954782, -51.2651672, 51.2409058
18: -22.0136814, 16.2653503, -22.0085487, 16.2981434, -32.9852219, 32.9706192
19: -22.1385593, 5.9775500, -22.1261063, 5.9923801, -21.8784752, 21.8994675
20: -27.8169212, 0.5465059, -27.8045959, 0.5595231, -22.2397461, 22.2863617
21: -26.2845039, 7.3222775, -26.2632332, 7.3352203, -28.7294312, 28.7698822
22: -29.4428043, 5.2141995, -29.4432964, 5.2191525, -27.3938522, 27.4099922
23: -17.8104286, 12.2332354, -17.8077278, 12.2617111, -23.9465027, 23.9544945
24: -16.4729233, 13.5775299, -16.4584103, 13.5842094, -25.7918930, 25.7483215
25: -23.8062706, 8.8100405, -23.8018646, 8.8100052, -27.8341217, 27.8549881
26: -39.2915421, 3.9604936, -39.3050079, 4.0475917, -33.1190109, 33.1313553
27: -19.4662495, 14.8063602, -19.4587498, 14.8116817, -34.2779312, 34.2651100
28: -21.3570404, 11.3734369, -21.3563881, 11.3951321, -26.1742783, 26.1979294
29: -24.3684444, 6.3402309, -24.3678284, 6.3519344, -28.5333328, 28.5298996
30: -30.0000687, 5.7931561, -29.9787750, 5.7945795, -33.8556976, 33.8693848
31: -23.3298950, 7.5928783, -23.3202038, 7.5976458, -24.9337311, 24.9548264
32: -36.9793816, -2.6778316, -36.9746132, -2.6917357, -29.7079010, 29.7258530
33: -54.3958931, -0.0462761, -54.4172211, -0.0794239, -44.3111572, 44.3038635
34: -49.1112633, -7.1025977, -49.1181221, -7.1101794, -35.7255325, 35.7162704
35: -40.2602005, 3.7880859, -40.2620735, 3.7585640, -35.8633728, 35.8787460
36: -45.5658188, 1.2068396, -45.5644646, 1.1786890, -39.5504608, 39.5782852
37: -60.6774292, -9.9593048, -60.6683044, -9.9555588, -44.8033600, 44.7973175
38: -53.5961533, 3.8973284, -53.5979691, 3.8665857, -48.6151123, 48.6415253
39: -61.8948135, -4.6615305, -61.8846703, -4.6893368, -48.1777039, 48.1920471
40: -50.1559906, -9.2082405, -50.1517258, -9.2035065, -39.0494995, 39.0150833
41: -31.9721756, 7.2209897, -31.9765453, 7.2124281, -37.8733215, 37.8833084
42: -30.0925751, -0.3745327, -30.1045322, -0.3723865, -23.0219879, 23.0320816

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7544873
time: 36.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7755462
time: 30.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.9408913, 25.9349041, -7.8207970, 25.8950768, -30.7320518, 30.6552467
1: -0.2953463, 26.8011837, -0.2222185, 26.7569809, -21.3021317, 21.2858429
2: -0.3077667, 25.8369102, -0.2366364, 25.7787781, -20.7551308, 20.7619820
3: -4.6435909, 22.5220699, -4.5878205, 22.4526253, -19.8103523, 19.8348618
4: -7.5672622, 22.5408363, -7.4741783, 22.4876785, -25.5265274, 25.5066643
5: -4.7343106, 24.8544559, -4.6680269, 24.7890339, -23.6577759, 23.6764984
6: -39.4169121, -4.2106762, -39.3769493, -4.2992258, -29.6398392, 29.6792603
7: -9.1306314, 23.4276829, -9.0517025, 23.3994865, -25.6652222, 25.6597672
8: -13.7841043, 20.0074940, -13.6633978, 19.9341316, -29.1296616, 29.1047668
9: -8.5563631, 22.5268059, -8.4854984, 22.4791946, -27.7601166, 27.7254257
10: -28.9609528, 17.8567619, -28.8462467, 17.7905731, -39.5363617, 39.5064011
11: -26.2297974, 6.7734528, -26.1397514, 6.7425723, -29.5907249, 29.5222626
12: -46.0889626, -8.4574432, -46.0114822, -8.5531654, -32.1558990, 32.1746750
13: -32.5670052, 13.2199059, -32.5694656, 13.1763439, -38.4296417, 38.4909592
14: -59.4258232, -2.2092152, -59.2614288, -2.2660675, -56.7017059, 56.6187592
15: -14.1637621, 18.8197460, -14.0770874, 18.7733612, -27.9356842, 27.8938751
16: -15.6317482, 22.2853947, -15.5685539, 22.2658520, -29.5722198, 29.5678139
17: -59.1868668, -6.9705114, -59.0634422, -6.9996939, -51.2639313, 51.2178040
18: -22.0381737, 16.2684956, -21.9512138, 16.2508507, -32.9937363, 32.9638824
19: -22.1515007, 5.9772778, -22.0710087, 5.9411211, -21.8885117, 21.8335762
20: -27.8327427, 0.5446367, -27.7520733, 0.4870400, -22.2441711, 22.2125435
21: -26.3010960, 7.3007059, -26.1943417, 7.2411542, -28.7199478, 28.6646118
22: -29.4747505, 5.2533760, -29.4247169, 5.2072635, -27.4106445, 27.3342056
23: -17.8229027, 12.2415028, -17.7563992, 12.2145710, -23.9571991, 23.9138947
24: -16.4866791, 13.6030140, -16.4441814, 13.5828800, -25.7779922, 25.7611313
25: -23.8287430, 8.8387051, -23.7773819, 8.7868309, -27.8635025, 27.8433533
26: -39.3283691, 3.9423666, -39.2236786, 3.8985100, -33.0907593, 33.0086212
27: -19.4826031, 14.8408527, -19.4344997, 14.8156862, -34.2982903, 34.2753525
28: -21.3723640, 11.3914690, -21.3042374, 11.3482380, -26.1916656, 26.1525345
29: -24.3928642, 6.3644466, -24.3369293, 6.3305745, -28.5320969, 28.4571609
30: -30.0086250, 5.7847610, -29.9430885, 5.7385983, -33.8491974, 33.8354568
31: -23.3439884, 7.6107903, -23.2663574, 7.5681686, -24.9625092, 24.9327621
32: -36.9896927, -2.6552706, -36.9565048, -2.7300072, -29.7010269, 29.7265778
33: -54.3948441, 0.0175638, -54.3357239, -0.1076174, -44.2878571, 44.3102188
34: -49.1379395, -7.0178084, -49.0813599, -7.1016245, -35.7694016, 35.7620926
35: -40.2921753, 3.8709841, -40.2366753, 3.7822838, -35.9120789, 35.9182281
36: -45.6133881, 1.2919559, -45.5539017, 1.1934166, -39.6021957, 39.6044846
37: -60.7157478, -9.9120045, -60.6397743, -9.9705515, -44.8218231, 44.7849884
38: -53.6607780, 4.0110979, -53.5747070, 3.8879251, -48.6856995, 48.7241058
39: -61.9285431, -4.6155119, -61.8735619, -4.7060070, -48.1890717, 48.2102432
40: -50.1698151, -9.1813011, -50.1276131, -9.2089291, -39.0367584, 39.0295486
41: -31.9925690, 7.2750626, -31.9546356, 7.2214575, -37.9067383, 37.9147949
42: -30.1046925, -0.3286648, -30.0920792, -0.3757319, -23.0350494, 23.0568657

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7356509
time: 35.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7567222
time: 33.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.9970303, 25.9358196, -7.9414310, 25.9150276, -30.8105774, 30.7546730
1: -0.3262815, 26.8023357, -0.2899656, 26.7819977, -21.3625565, 21.3526154
2: -0.3663669, 25.8388691, -0.3569207, 25.8192940, -20.8543205, 20.8520279
3: -4.7094259, 22.5241394, -4.7217417, 22.5031509, -19.9343987, 19.9079819
4: -7.6208887, 22.5437069, -7.5862484, 22.5303555, -25.6201401, 25.5764885
5: -4.8067427, 24.8583050, -4.8161545, 24.8392982, -23.7842102, 23.7671089
6: -39.4232407, -4.2020979, -39.3968239, -4.2765408, -29.6674423, 29.7122192
7: -9.1740894, 23.4300594, -9.1505299, 23.4151039, -25.7484665, 25.7675400
8: -13.8293228, 20.0115242, -13.7582607, 19.9784622, -29.2215195, 29.1967850
9: -8.5656967, 22.5479355, -8.5129299, 22.5265312, -27.8155594, 27.8113747
10: -28.9668884, 17.9096699, -28.8961163, 17.9042778, -39.6407242, 39.6064606
11: -26.2334538, 6.8169389, -26.2195396, 6.8306513, -29.5883675, 29.6571045
12: -46.0920105, -8.3438549, -46.0868454, -8.3211699, -32.3292694, 32.3624268
13: -32.5753326, 13.2349691, -32.5886803, 13.2166138, -38.4682770, 38.5761185
14: -59.4447937, -2.1135883, -59.3754654, -2.0725975, -56.8357391, 56.8273773
15: -14.1863327, 18.8221321, -14.1282825, 18.7956238, -27.9830170, 27.9226608
16: -15.6458588, 22.2946949, -15.6162186, 22.2875252, -29.5975876, 29.6269684
17: -59.1940384, -6.9164276, -59.1534691, -6.8869877, -51.3418427, 51.3585968
18: -22.0434456, 16.3023987, -22.0171089, 16.3165417, -33.0347443, 33.0127640
19: -22.1607914, 6.0116339, -22.1324310, 6.0098472, -21.9204369, 21.9351234
20: -27.8403282, 0.5914865, -27.8081989, 0.5820751, -22.2860794, 22.3280525
21: -26.3097496, 7.3557954, -26.2720356, 7.3521523, -28.7735291, 28.8098640
22: -29.4813900, 5.2702551, -29.4501686, 5.2482271, -27.4623184, 27.4440536
23: -17.8306618, 12.2742786, -17.8121338, 12.2818909, -23.9887695, 23.9874802
24: -16.4928513, 13.6116142, -16.4626789, 13.6013966, -25.8290482, 25.7781296
25: -23.8328457, 8.8625870, -23.8052902, 8.8368950, -27.8887558, 27.8860626
26: -39.3334198, 4.0371909, -39.3102455, 4.0864515, -33.2000275, 33.1896133
27: -19.4956894, 14.8466053, -19.4687538, 14.8317127, -34.3274002, 34.3153610
28: -21.3833160, 11.4284277, -21.3607025, 11.4234848, -26.2316360, 26.2427597
29: -24.3997898, 6.3838968, -24.3764381, 6.3745170, -28.5882339, 28.5643425
30: -30.0121918, 5.8151054, -29.9837494, 5.8045321, -33.8841248, 33.9003067
31: -23.3553524, 7.6359138, -23.3273087, 7.6192060, -24.9840775, 25.0059547
32: -36.9944382, -2.6250067, -36.9788818, -2.6654501, -29.7613220, 29.7851562
33: -54.4371796, 0.0290146, -54.4230881, -0.0416546, -44.3943024, 44.3465118
34: -49.1557388, -7.0119457, -49.1214294, -7.0651622, -35.8155670, 35.7900467
35: -40.3059883, 3.8728733, -40.2664261, 3.8014698, -35.9508972, 35.9361496
36: -45.6181221, 1.3044834, -45.5692596, 1.2278118, -39.6524353, 39.6651230
37: -60.7226295, -9.8883390, -60.6743546, -9.9191227, -44.8873901, 44.8355865
38: -53.6670151, 4.0282431, -53.6047440, 3.9314575, -48.7533264, 48.7567368
39: -61.9370346, -4.6009779, -61.8949814, -4.6591196, -48.2514191, 48.2519302
40: -50.1794930, -9.1761589, -50.1602402, -9.1879482, -39.0943909, 39.0587845
41: -32.0015106, 7.2845268, -31.9813614, 7.2441168, -37.9420242, 37.9517136
42: -30.1097488, -0.3148956, -30.1071281, -0.3425827, -23.0648460, 23.0921135

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7549967
time: 25.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7760702
time: 35.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.9112053, 25.9146252, -7.9618053, 25.9182243, -30.7276077, 30.7726860
1: -0.2734356, 26.7843018, -0.3078127, 26.7802429, -21.3164673, 21.3387947
2: -0.2982290, 25.8176899, -0.3236384, 25.8023109, -20.7871704, 20.8077049
3: -4.6355524, 22.5142002, -4.6557851, 22.4786644, -19.8437843, 19.8754578
4: -7.5457683, 22.5290051, -7.5795918, 22.5063782, -25.5415001, 25.5713768
5: -4.7153988, 24.8352985, -4.7360592, 24.8106995, -23.6801682, 23.7042160
6: -39.4096451, -4.2095251, -39.4110832, -4.1808071, -29.7299423, 29.7131042
7: -9.0845995, 23.3992653, -9.1132536, 23.4168415, -25.6692429, 25.6709633
8: -13.7867804, 19.9896431, -13.8095808, 19.9739990, -29.1888008, 29.1949158
9: -8.5208101, 22.5002823, -8.5612803, 22.5064621, -27.7737579, 27.7588234
10: -28.9501400, 17.8336639, -28.9507523, 17.8364716, -39.6040649, 39.5650864
11: -26.1991520, 6.7930608, -26.1486950, 6.8151245, -29.6193237, 29.5516815
12: -46.0763626, -8.4934521, -46.0222015, -8.4922848, -32.2221222, 32.1454926
13: -32.5292397, 13.2161608, -32.5763931, 13.2144032, -38.5175171, 38.4705811
14: -59.4159317, -2.2313843, -59.3809853, -2.2198114, -56.7369995, 56.6885071
15: -14.1923065, 18.8095741, -14.2067413, 18.8009529, -27.9893799, 28.0017204
16: -15.5743408, 22.2464962, -15.6051788, 22.2851181, -29.5793610, 29.5221596
17: -59.1663322, -6.9840689, -59.1255646, -6.9689703, -51.2986450, 51.2034607
18: -22.0233269, 16.2618980, -21.9892445, 16.2956238, -33.0360184, 33.0364685
19: -22.1386795, 5.9896631, -22.1105938, 6.0193653, -21.9412155, 21.8865318
20: -27.8156090, 0.5540867, -27.7936707, 0.5768461, -22.2996063, 22.2691574
21: -26.2884445, 7.3296547, -26.2451954, 7.3532596, -28.8038254, 28.7452812
22: -29.4429855, 5.2403955, -29.4636440, 5.2652836, -27.4108276, 27.3946762
23: -17.8095188, 12.2351561, -17.7842598, 12.2669907, -23.9798813, 23.9468460
24: -16.4731541, 13.6077394, -16.4811554, 13.6452770, -25.8170700, 25.8106079
25: -23.8073769, 8.8408728, -23.8108902, 8.8748407, -27.9054565, 27.9139252
26: -39.2940598, 3.9093585, -39.2544746, 3.9472342, -33.0992203, 33.0353165
27: -19.4664917, 14.8461256, -19.4759045, 14.8900185, -34.3565102, 34.3220291
28: -21.3527260, 11.3911171, -21.3390884, 11.4328785, -26.2425232, 26.2011414
29: -24.3685379, 6.3524146, -24.3671875, 6.3743534, -28.5281830, 28.4975739
30: -30.0025883, 5.8201981, -29.9789104, 5.8495526, -33.9423370, 33.9034805
31: -23.3283577, 7.6210804, -23.3062687, 7.6574955, -25.0255737, 24.9787407
32: -36.9811707, -2.6596074, -36.9800987, -2.6539183, -29.7451324, 29.7471924
33: -54.3637733, 0.0187588, -54.3971214, 0.0114517, -44.3185883, 44.4156418
34: -49.1022415, -7.0530434, -49.1283150, -7.0334778, -35.7889328, 35.7945557
35: -40.2556305, 3.8557434, -40.2920990, 3.8810205, -35.9427795, 35.9931641
36: -45.5676575, 1.2724562, -45.6093712, 1.3039598, -39.6221161, 39.6562271
37: -60.6816635, -9.9334126, -60.6970749, -9.9053822, -44.8068237, 44.8609009
38: -53.6011810, 3.9777699, -53.6461792, 4.0239582, -48.6921692, 48.7808685
39: -61.8969727, -4.6096563, -61.9330292, -4.5999336, -48.2129364, 48.2909927
40: -50.1595383, -9.1979570, -50.1577148, -9.1913233, -39.0373230, 39.0590363
41: -31.9714298, 7.2576799, -31.9862480, 7.2867007, -37.9423828, 37.9317093
42: -30.0927887, -0.3552995, -30.1022720, -0.3349152, -23.0743675, 23.0515289

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7667740
time: 36.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7878132
time: 33.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7861191
time: 34.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7498560, upper bound: 13.8071613
time: 33.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.0374231, 25.9379654, -8.0248709, 25.9201126, -30.8547478, 30.8599777
1: -0.3571882, 26.8032494, -0.3497248, 26.7809925, -21.3878670, 21.4005051
2: -0.3666177, 25.8390865, -0.3584232, 25.8034077, -20.8394012, 20.8636322
3: -4.6903477, 22.5283890, -4.6839147, 22.4807701, -19.8866310, 19.9215698
4: -7.6406779, 22.5439682, -7.6270895, 22.5077648, -25.6174965, 25.6360893
5: -4.7843518, 24.8577862, -4.7715359, 24.8132381, -23.7334137, 23.7641144
6: -39.4225769, -4.1386213, -39.4151726, -4.1459646, -29.7966919, 29.7896500
7: -9.1854830, 23.4295712, -9.1647425, 23.4183998, -25.7419586, 25.7543831
8: -13.8774090, 20.0122108, -13.8561697, 19.9769135, -29.2651138, 29.2650528
9: -8.6144056, 22.5314407, -8.6072311, 22.5079041, -27.8486481, 27.8369064
10: -29.0316849, 17.8660393, -28.9922409, 17.8412018, -39.6652603, 39.6371689
11: -26.2382488, 6.8122001, -26.1680450, 6.8239484, -29.6762466, 29.5906410
12: -46.0953178, -8.4102459, -46.0257072, -8.4512339, -32.2779312, 32.2401085
13: -32.5807533, 13.2389793, -32.6019821, 13.2224550, -38.5804901, 38.5494766
14: -59.5026550, -2.2009487, -59.4235191, -2.2160444, -56.8325195, 56.7893524
15: -14.2335091, 18.8277149, -14.2252607, 18.8109932, -28.0421753, 28.0451736
16: -15.6729088, 22.2871380, -15.6532784, 22.2862759, -29.6392517, 29.6124268
17: -59.2321014, -6.9605656, -59.1582031, -6.9604826, -51.3753128, 51.3211823
18: -22.0530071, 16.2989826, -21.9977646, 16.3140278, -33.0855560, 33.0786362
19: -22.1609440, 6.0237713, -22.1168823, 6.0368023, -21.9831657, 21.9222069
20: -27.8390160, 0.5990291, -27.7972527, 0.5994158, -22.3459244, 22.3108292
21: -26.3136711, 7.3632069, -26.2539482, 7.3702488, -28.8478775, 28.7852631
22: -29.4815712, 5.2964697, -29.4704552, 5.2943621, -27.4792252, 27.4286957
23: -17.8297691, 12.2762146, -17.7886963, 12.2872066, -24.0221252, 23.9798088
24: -16.4930706, 13.6418056, -16.4853916, 13.6624451, -25.8542328, 25.8404312
25: -23.8339691, 8.8934193, -23.8142433, 8.9017019, -27.9600601, 27.9449539
26: -39.3358803, 3.9860153, -39.2597237, 3.9861102, -33.1801758, 33.0935669
27: -19.4959621, 14.8863487, -19.4858456, 14.9100761, -34.4060364, 34.3721924
28: -21.3789711, 11.4461212, -21.3433933, 11.4612637, -26.2998962, 26.2459717
29: -24.3998375, 6.3960986, -24.3758011, 6.3969679, -28.5830688, 28.5320625
30: -30.0147495, 5.8421497, -29.9838943, 5.8595276, -33.9708252, 33.9344025
31: -23.3538208, 7.6641321, -23.3133488, 7.6790247, -25.0759735, 25.0298271
32: -36.9961891, -2.6067495, -36.9843674, -2.6275826, -29.7986374, 29.8065338
33: -54.4049873, 0.0940418, -54.4030457, 0.0492449, -44.4017944, 44.4582596
34: -49.1467094, -6.9623303, -49.1315727, -6.9883447, -35.8789978, 35.8683624
35: -40.3014488, 3.9405041, -40.2964745, 3.9239407, -36.0303345, 36.0505600
36: -45.6199875, 1.3700438, -45.6140518, 1.3531284, -39.7241669, 39.7430038
37: -60.7268486, -9.8625164, -60.7031364, -9.8689804, -44.8908997, 44.8991470
38: -53.6720390, 4.1086845, -53.6528397, 4.0888577, -48.8302917, 48.8960266
39: -61.9391632, -4.5490055, -61.9433136, -4.5697050, -48.2866058, 48.3508377
40: -50.1830902, -9.1658697, -50.1662216, -9.1757469, -39.0822754, 39.1027222
41: -32.0007515, 7.3212190, -31.9910450, 7.3183517, -38.0109711, 38.0000992
42: -30.1099415, -0.2956514, -30.1048241, -0.3050756, -23.1172523, 23.1115761

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7673265
time: 29.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7883757
time: 28.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2

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

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7866707
time: 33.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7714884, upper bound: 13.8077234
time: 30.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.9315147, 25.9728470, -7.7971306, 25.8940163, -30.7078476, 30.6732941
1: -0.2860675, 26.8571568, -0.2035661, 26.7572994, -21.2979507, 21.3233490
2: -0.3468494, 25.9241924, -0.2431641, 25.7797585, -20.7988434, 20.8558502
3: -4.7097192, 22.6262169, -4.6077051, 22.4511490, -19.8479843, 19.9096985
4: -7.5686688, 22.6223183, -7.4623613, 22.4887772, -25.5352135, 25.5749626
5: -4.7908454, 24.9419403, -4.6812696, 24.7877846, -23.7014389, 23.7654419
6: -39.4333916, -4.2608252, -39.3773155, -4.3337383, -29.6120682, 29.6279678
7: -9.1195564, 23.4475307, -9.0269871, 23.3990479, -25.6754494, 25.6514359
8: -13.7583084, 20.0704231, -13.6396608, 19.9347630, -29.1195221, 29.1454086
9: -8.4873819, 22.5173149, -8.4460545, 22.4802361, -27.7568283, 27.6567688
10: -28.9519882, 17.9107742, -28.8083000, 17.8104248, -39.5952530, 39.5250702
11: -26.4191933, 6.8647256, -26.1236153, 6.7778730, -29.8095016, 29.5640717
12: -46.1907425, -8.3847866, -46.0098915, -8.5340309, -32.2765121, 32.2060509
13: -32.5475807, 13.2872295, -32.5506859, 13.1755342, -38.4083405, 38.5073395
14: -59.5542755, -2.0838718, -59.2312775, -2.2053680, -56.8752441, 56.6345673
15: -14.1557999, 18.8495235, -14.0583315, 18.7649517, -27.9340210, 27.8893433
16: -15.6183624, 22.2580032, -15.5309029, 22.2580624, -29.6013336, 29.5007401
17: -59.3573875, -6.8834438, -59.0368462, -6.9634571, -51.4765167, 51.1814575
18: -22.1844616, 16.3132267, -21.9460030, 16.2668076, -33.0943680, 32.9681168
19: -22.2867985, 6.0313988, -22.0707130, 5.9605026, -22.0444221, 21.8677597
20: -27.9345284, 0.5930285, -27.7535515, 0.5025272, -22.3717232, 22.2448883
21: -26.4738369, 7.3869367, -26.1922016, 7.2737470, -28.9280243, 28.7257462
22: -29.5347176, 5.2483282, -29.4230804, 5.1961031, -27.4652710, 27.3545876
23: -17.9447937, 12.2893562, -17.7570381, 12.2293825, -24.0948639, 23.9518623
24: -16.5492878, 13.6293306, -16.4428558, 13.5905476, -25.8423386, 25.7743835
25: -23.8894691, 8.8525591, -23.7768784, 8.7859974, -27.9192200, 27.8627090
26: -39.4937134, 4.0314741, -39.2220612, 3.9303346, -33.2989426, 33.0801086
27: -19.5573082, 14.8654671, -19.4321156, 14.8238754, -34.3811836, 34.2975845
28: -21.4949703, 11.4369993, -21.3073769, 11.3609829, -26.3260803, 26.1899414
29: -24.4873829, 6.3727241, -24.3340321, 6.3265309, -28.6446228, 28.4791412
30: -30.1558361, 5.8744240, -29.9416809, 5.7705564, -34.0211945, 33.8986435
31: -23.4688492, 7.6437416, -23.2662849, 7.5782170, -25.0981674, 24.9389458
32: -36.9980278, -2.6736774, -36.9561348, -2.7497101, -29.6834259, 29.7058182
33: -54.4365349, 0.0828304, -54.3628159, -0.1368942, -44.2677917, 44.4561234
34: -49.1185722, -7.0681591, -49.0818214, -7.1427851, -35.7449417, 35.7187119
35: -40.2623634, 3.8232670, -40.2340317, 3.7407093, -35.8573914, 35.9167175
36: -45.5869408, 1.2261400, -45.5522690, 1.1491346, -39.5610809, 39.5548630
37: -60.7239189, -9.9450665, -60.6363754, -9.9988308, -44.7747192, 44.8136673
38: -53.6274757, 3.9187756, -53.5712852, 3.8261051, -48.5661926, 48.7052536
39: -61.9234619, -4.5718689, -61.8759766, -4.7262325, -48.1566162, 48.2756805
40: -50.1848984, -9.1701746, -50.1231995, -9.2212210, -39.0201569, 39.0547485
41: -31.9964828, 7.2403560, -31.9554787, 7.1926870, -37.8759460, 37.8806610
42: -30.1089668, -0.3553314, -30.0931549, -0.4026341, -23.0273132, 23.0299988

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7351426
time: 39.06 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7561999
time: 35.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.9881430, 25.9737930, -7.9183712, 25.9140072, -30.7869186, 30.7733574
1: -0.3171806, 26.8583107, -0.2717934, 26.7823238, -21.3586998, 21.3905373
2: -0.4059887, 25.9261665, -0.3640904, 25.8203163, -20.8988190, 20.9465332
3: -4.7760115, 22.6282005, -4.7421298, 22.5016594, -19.9723663, 19.9832306
4: -7.6226988, 22.6252766, -7.5749702, 22.5315285, -25.6292343, 25.6454201
5: -4.8637714, 24.9457054, -4.8299642, 24.8381538, -23.8284912, 23.8565788
6: -39.4398041, -4.2521868, -39.3972588, -4.3109312, -29.6395264, 29.6610489
7: -9.1649466, 23.4499550, -9.1278868, 23.4147301, -25.7592316, 25.7598572
8: -13.8040495, 20.0744934, -13.7351465, 19.9791355, -29.2118568, 29.2380753
9: -8.4966030, 22.5386066, -8.4733925, 22.5280476, -27.8126144, 27.7429962
10: -28.9579811, 17.9642239, -28.8582802, 17.9247665, -39.7002716, 39.6258240
11: -26.4228821, 6.9105968, -26.2034664, 6.8689976, -29.8072281, 29.6991692
12: -46.1938171, -8.2699299, -46.0852890, -8.3006897, -32.4512024, 32.3949776
13: -32.5568733, 13.3026485, -32.5699539, 13.2163019, -38.4477386, 38.5827637
14: -59.5733032, -1.9874468, -59.3456192, -2.0109768, -57.0066376, 56.8442688
15: -14.1783409, 18.8519726, -14.1096420, 18.7872677, -27.9813766, 27.9126434
16: -15.6324577, 22.2673111, -15.5786781, 22.2798367, -29.6263275, 29.5601158
17: -59.3646927, -6.8289642, -59.1270485, -6.8502464, -51.5564575, 51.3228760
18: -22.1897507, 16.3472023, -22.0120163, 16.3325710, -33.1235886, 33.0179367
19: -22.2961235, 6.0657849, -22.1321621, 6.0293069, -22.0765533, 21.9694214
20: -27.9421082, 0.6400905, -27.8097153, 0.5978398, -22.4121857, 22.3607674
21: -26.4825172, 7.4421706, -26.2699795, 7.3850536, -28.9820099, 28.8713188
22: -29.5414238, 5.2664003, -29.4485741, 5.2373943, -27.5173035, 27.4645576
23: -17.9526138, 12.3220615, -17.8129292, 12.2967300, -24.1264648, 24.0256844
24: -16.5556526, 13.6379137, -16.4616718, 13.6090670, -25.8789291, 25.7943039
25: -23.8936119, 8.8766909, -23.8047829, 8.8362637, -27.9444199, 27.9058762
26: -39.4989929, 4.1268048, -39.3087234, 4.1189270, -33.4052734, 33.2620239
27: -19.5702133, 14.8742046, -19.4668274, 14.8408041, -34.4110184, 34.3410339
28: -21.5059471, 11.4740458, -21.3639107, 11.4363880, -26.3661423, 26.2802505
29: -24.4943695, 6.3934679, -24.3735790, 6.3717942, -28.7010803, 28.5866547
30: -30.1594620, 5.9046855, -29.9824104, 5.8365288, -34.0562592, 33.9634705
31: -23.4801788, 7.6688547, -23.3272343, 7.6292419, -25.1195526, 25.0121002
32: -37.0027466, -2.6432028, -36.9784851, -2.6845980, -29.7442780, 29.7644806
33: -54.4789658, 0.0943899, -54.4503937, -0.0708866, -44.3753052, 44.4913101
34: -49.1368179, -7.0623102, -49.1226730, -7.1062832, -35.7873611, 35.7452927
35: -40.2761879, 3.8251047, -40.2638931, 3.7599573, -35.8938675, 35.9256287
36: -45.5917549, 1.2390661, -45.5676193, 1.1838455, -39.6123886, 39.6154709
37: -60.7307129, -9.9211597, -60.6709251, -9.9469090, -44.8401489, 44.8641891
38: -53.6338501, 3.9358921, -53.6013985, 3.8699446, -48.6304474, 48.7351685
39: -61.9319344, -4.5571976, -61.8973274, -4.6792116, -48.2192383, 48.3151779
40: -50.1948013, -9.1648178, -50.1559563, -9.2002020, -39.0779114, 39.0842819
41: -32.0056190, 7.2498698, -31.9823589, 7.2153950, -37.9112625, 37.9180222
42: -30.1141491, -0.3414307, -30.1082478, -0.3693228, -23.0545502, 23.0662117

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7544873
time: 34.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7755462
time: 34.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.0577269, 25.9962177, -7.8602009, 25.8959122, -30.8349228, 30.7605553
1: -0.3698254, 26.8761253, -0.2454233, 26.7580338, -21.3693390, 21.3850441
2: -0.4152308, 25.9455948, -0.2779768, 25.7808762, -20.8510284, 20.9117088
3: -4.7645125, 22.6404457, -4.6357703, 22.4532318, -19.8908463, 19.9558220
4: -7.6635280, 22.6373024, -7.5098000, 22.4901390, -25.6111526, 25.6396561
5: -4.8598061, 24.9644470, -4.7167625, 24.7903290, -23.7547379, 23.8253479
6: -39.4463196, -4.1899853, -39.3814163, -4.2989302, -29.6787567, 29.7044678
7: -9.2204409, 23.4778175, -9.0784492, 23.4006729, -25.7481842, 25.7348099
8: -13.8489313, 20.0930462, -13.6862078, 19.9376831, -29.1958084, 29.2155342
9: -8.5809889, 22.5484924, -8.4920549, 22.4817047, -27.8316269, 27.7348518
10: -29.0335388, 17.9431591, -28.8498077, 17.8151398, -39.6563950, 39.5972061
11: -26.4582520, 6.8838987, -26.1429329, 6.7866573, -29.8663788, 29.6029968
12: -46.2096901, -8.3015575, -46.0133514, -8.4930420, -32.3322296, 32.3005676
13: -32.5990906, 13.3099775, -32.5763321, 13.1835308, -38.4713287, 38.5862808
14: -59.6408653, -2.0534668, -59.2738495, -2.2015905, -56.9707489, 56.7354889
15: -14.1968575, 18.8676929, -14.0767365, 18.7749958, -27.9867554, 27.9327583
16: -15.7168674, 22.2986546, -15.5790138, 22.2591782, -29.6612396, 29.5909615
17: -59.4232178, -6.8599586, -59.0694084, -6.9549704, -51.5531921, 51.2992249
18: -22.2142105, 16.3502407, -21.9545994, 16.2852249, -33.1439743, 33.0102844
19: -22.3090591, 6.0654893, -22.0769901, 5.9779320, -22.0863724, 21.9034424
20: -27.9579086, 0.6380186, -27.7571259, 0.5250735, -22.4180374, 22.2865982
21: -26.4990425, 7.4204926, -26.2009983, 7.2907162, -28.9720840, 28.7657509
22: -29.5733242, 5.3043723, -29.4299583, 5.2251315, -27.5337143, 27.3886719
23: -17.9650440, 12.3304195, -17.7614689, 12.2495670, -24.1370697, 23.9848289
24: -16.5692406, 13.6633949, -16.4471302, 13.6076984, -25.8795929, 25.8042374
25: -23.9160748, 8.9051361, -23.7802486, 8.8128910, -27.9738770, 27.8937759
26: -39.5356827, 4.1081982, -39.2273064, 3.9692276, -33.3799286, 33.1383286
27: -19.5867805, 14.9057045, -19.4421577, 14.8439093, -34.4306908, 34.3478622
28: -21.5211754, 11.4920378, -21.3116837, 11.3893824, -26.3833771, 26.2347488
29: -24.5186844, 6.4163675, -24.3426552, 6.3490934, -28.6994324, 28.5135918
30: -30.1679611, 5.8963413, -29.9466724, 5.7805042, -34.0495529, 33.9295425
31: -23.4943199, 7.6868277, -23.2733574, 7.5997415, -25.1485901, 24.9900284
32: -37.0130348, -2.6208897, -36.9604416, -2.7234077, -29.7369003, 29.7651138
33: -54.4777756, 0.1581411, -54.3686943, -0.0991249, -44.3509827, 44.4986572
34: -49.1631241, -6.9774957, -49.0851479, -7.0977225, -35.8349915, 35.7925110
35: -40.3082047, 3.9079876, -40.2384491, 3.7836275, -35.9449310, 35.9740906
36: -45.6392860, 1.3237925, -45.5569611, 1.1982632, -39.6630936, 39.6416473
37: -60.7691612, -9.8741055, -60.6423988, -9.9623556, -44.8587646, 44.8519058
38: -53.6983452, 4.0496264, -53.5779457, 3.8909740, -48.7044678, 48.8204193
39: -61.9656754, -4.5112772, -61.8862839, -4.6960278, -48.2303162, 48.3355560
40: -50.2084427, -9.1380968, -50.1317215, -9.2056561, -39.0650787, 39.0984192
41: -32.0258102, 7.3039179, -31.9602928, 7.2242961, -37.9446259, 37.9490204
42: -30.1261387, -0.2957220, -30.0957451, -0.3728762, -23.0701752, 23.0900040

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7356509
time: 34.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7567222
time: 34.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1143856, 25.9971390, -7.9814224, 25.9159184, -30.9140167, 30.8606186
1: -0.4009061, 26.8772564, -0.3136439, 26.7831059, -21.4300652, 21.4522285
2: -0.4743690, 25.9475746, -0.3989019, 25.8213997, -20.9510269, 21.0024338
3: -4.8307762, 22.6424484, -4.7702522, 22.5037346, -20.0152435, 20.0293579
4: -7.7175636, 22.6402454, -7.6223488, 22.5329075, -25.7051888, 25.7100868
5: -4.9327593, 24.9682407, -4.8654609, 24.8406925, -23.8817940, 23.9165306
6: -39.4527817, -4.1813107, -39.4013481, -4.2760830, -29.7062225, 29.7375717
7: -9.2658730, 23.4802704, -9.1794081, 23.4163132, -25.8319283, 25.8431892
8: -13.8946371, 20.0971260, -13.7817259, 19.9820576, -29.2881699, 29.3081551
9: -8.5902319, 22.5697479, -8.5192995, 22.5294971, -27.8874512, 27.8210526
10: -29.0395432, 17.9966583, -28.8997955, 17.9295006, -39.7614517, 39.6979599
11: -26.4619999, 6.9297404, -26.2228432, 6.8777804, -29.8640671, 29.7380905
12: -46.2128143, -8.1867256, -46.0887680, -8.2596798, -32.5069580, 32.4894829
13: -32.6083755, 13.3254652, -32.5955048, 13.2243099, -38.5107269, 38.6617050
14: -59.6599388, -1.9570160, -59.3881264, -2.0071821, -57.1020966, 56.9450836
15: -14.2193851, 18.8701191, -14.1280441, 18.7973232, -28.0341034, 27.9560318
16: -15.7309685, 22.3079605, -15.6267776, 22.2809467, -29.6862183, 29.6503334
17: -59.4304924, -6.8054543, -59.1595573, -6.8417015, -51.6331024, 51.4405670
18: -22.2195263, 16.3842926, -22.0205345, 16.3510113, -33.1732101, 33.0600662
19: -22.3183498, 6.0998964, -22.1384964, 6.0467844, -22.1184959, 22.0051193
20: -27.9655037, 0.6850991, -27.8133354, 0.6204267, -22.4584808, 22.4024620
21: -26.5077629, 7.4756732, -26.2787132, 7.4019685, -29.0261002, 28.9113159
22: -29.5800095, 5.3224478, -29.4554138, 5.2663989, -27.5857315, 27.4985809
23: -17.9728374, 12.3631220, -17.8173065, 12.3169413, -24.1686859, 24.0586510
24: -16.5756035, 13.6719952, -16.4659004, 13.6261940, -25.9161301, 25.8241653
25: -23.9202175, 8.9292126, -23.8082619, 8.8631802, -27.9990540, 27.9369545
26: -39.5408363, 4.2035499, -39.3139725, 4.1577697, -33.4862671, 33.3202820
27: -19.5997162, 14.9144516, -19.4767742, 14.8608570, -34.4605713, 34.3912277
28: -21.5321541, 11.5290117, -21.3682480, 11.4647627, -26.4234619, 26.3250275
29: -24.5256844, 6.4371576, -24.3822384, 6.3944092, -28.7559357, 28.6211205
30: -30.1715965, 5.9266400, -29.9873772, 5.8464899, -34.0846939, 33.9943619
31: -23.5056705, 7.7118511, -23.3343430, 7.6507912, -25.1699142, 25.0631943
32: -37.0177765, -2.5903549, -36.9827690, -2.6583309, -29.7977142, 29.8237915
33: -54.5201874, 0.1696491, -54.4562607, -0.0331059, -44.4585419, 44.5338821
34: -49.1813736, -6.9716773, -49.1259727, -7.0611820, -35.8773804, 35.8190765
35: -40.3220329, 3.9099064, -40.2682724, 3.8028269, -35.9813690, 35.9829712
36: -45.6440926, 1.3366919, -45.5723419, 1.2329445, -39.7144089, 39.7022858
37: -60.7758942, -9.8502369, -60.6770668, -9.9104538, -44.9242249, 44.9023743
38: -53.7047844, 4.0667353, -53.6081390, 3.9348125, -48.7686768, 48.8504410
39: -61.9742203, -4.4965591, -61.9076157, -4.6491108, -48.2929840, 48.3750916
40: -50.2183533, -9.1328039, -50.1644440, -9.1846218, -39.1228180, 39.1278915
41: -32.0349426, 7.3134060, -31.9871368, 7.2470860, -37.9799500, 37.9863434
42: -30.1313000, -0.2818103, -30.1108246, -0.3395228, -23.0974274, 23.1262703

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7549967
time: 59.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7760702
time: 36.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0280685, 25.9759254, -8.0012112, 25.9190712, -30.8305016, 30.8780365
1: -0.3479056, 26.8592110, -0.3310680, 26.7813072, -21.3836403, 21.4379921
2: -0.4056740, 25.9263458, -0.3649435, 25.8044243, -20.8830948, 20.9574738
3: -4.7564955, 22.6325588, -4.7037787, 22.4792519, -19.9242744, 19.9964371
4: -7.6420927, 22.6254578, -7.6152287, 22.5088406, -25.6261444, 25.7043686
5: -4.8409104, 24.9453106, -4.7847629, 24.8120193, -23.7770767, 23.8530273
6: -39.4390030, -4.1888361, -39.4155235, -4.1805277, -29.7688599, 29.7383423
7: -9.1744194, 23.4494362, -9.1400604, 23.4179497, -25.7522125, 25.7460327
8: -13.8515749, 20.0751572, -13.8324518, 19.9775429, -29.2548904, 29.3056793
9: -8.5454493, 22.5219269, -8.5678511, 22.5089588, -27.8452988, 27.7682571
10: -29.0226917, 17.9201355, -28.9543190, 17.8610363, -39.7241287, 39.6559296
11: -26.4276581, 6.9034781, -26.1518230, 6.8592105, -29.8949280, 29.6324844
12: -46.1971512, -8.3374996, -46.0241470, -8.4320517, -32.3985519, 32.2714500
13: -32.5614243, 13.3062782, -32.5832748, 13.2216406, -38.5592575, 38.5659180
14: -59.6309814, -2.0756187, -59.3933830, -2.1553116, -57.0060272, 56.8052368
15: -14.2254162, 18.8575211, -14.2064238, 18.8026028, -28.0404434, 28.0405998
16: -15.6594715, 22.2596951, -15.6156235, 22.2784519, -29.6683731, 29.5453033
17: -59.4025879, -6.8735304, -59.1315727, -6.9242086, -51.5878372, 51.2848358
18: -22.1992989, 16.3437481, -21.9925728, 16.3300171, -33.1863022, 33.0828476
19: -22.2962589, 6.0779219, -22.1166134, 6.0562754, -22.1390724, 21.9563713
20: -27.9407654, 0.6474524, -27.7987099, 0.6149054, -22.4734802, 22.3431931
21: -26.4864616, 7.4494033, -26.2518501, 7.4028444, -29.0559921, 28.8464584
22: -29.5415955, 5.2914128, -29.4688301, 5.2831721, -27.5338974, 27.4491158
23: -17.9516411, 12.3240881, -17.7893486, 12.3019562, -24.1597366, 24.0177536
24: -16.5556374, 13.6681261, -16.4840775, 13.6701250, -25.9186630, 25.8536911
25: -23.8947029, 8.9073305, -23.8137341, 8.9009056, -28.0157928, 27.9643860
26: -39.5013351, 4.0751643, -39.2580910, 4.0179329, -33.3883820, 33.1650543
27: -19.5707150, 14.9109974, -19.4835396, 14.9182959, -34.4890099, 34.3945389
28: -21.5015259, 11.4916801, -21.3465252, 11.4740028, -26.4342499, 26.2833481
29: -24.4944038, 6.4043756, -24.3728561, 6.3928976, -28.6955566, 28.5540009
30: -30.1619377, 5.9318151, -29.9824696, 5.8914680, -34.1427002, 33.9975662
31: -23.4786644, 7.6971245, -23.3132992, 7.6890478, -25.2116241, 25.0360222
32: -37.0045433, -2.6252632, -36.9839935, -2.6472735, -29.7810516, 29.7857742
33: -54.4467125, 0.1592999, -54.4300880, 0.0199642, -44.3817291, 44.6040955
34: -49.1274376, -7.0127182, -49.1320801, -7.0295334, -35.8545227, 35.8249969
35: -40.2716713, 3.8927460, -40.2938080, 3.8823633, -35.9757080, 36.0491333
36: -45.5935898, 1.3042812, -45.6123848, 1.3088617, -39.6831207, 39.6933746
37: -60.7350311, -9.8955421, -60.6997070, -9.8971977, -44.8436890, 44.9278259
38: -53.6387634, 4.0163975, -53.6495132, 4.0269680, -48.7108917, 48.8772583
39: -61.9341660, -4.5054016, -61.9456902, -4.5900192, -48.2542419, 48.4162521
40: -50.1981659, -9.1546907, -50.1618233, -9.1880264, -39.0656738, 39.1279297
41: -32.0047073, 7.2865291, -31.9919052, 7.2895322, -37.9801636, 37.9659348
42: -30.1142445, -0.3223338, -30.1059322, -0.3320460, -23.1095047, 23.0847015

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7667739
time: 32.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7878132
time: 31.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

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

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7861191
time: 40.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7860934, upper bound: 13.8071613
time: 32.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

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

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7673265
time: 32.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7883757
time: 33.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2

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

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7866707
time: 91.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.8077234
time: 37.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 130.75 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7351426
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7561999
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7544873
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7755462
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7356509
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7567222
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7549967
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7709627, upper bound: 13.7760702
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7667740
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7878132
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7498560, upper bound: 13.7861191
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7498560, upper bound: 13.8071613
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7673265
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7883757
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7866707
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7714884, upper bound: 13.8077234
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7351426
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7561999
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7544873
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7755462
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7356509
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7567222
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7549967
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7760702
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7667739
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7878132
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7861191
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.7860934, upper bound: 13.8071613
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7673265
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7883757
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7866707
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 130.75
Output dim: 1, lower bound: -13.8077233, upper bound: 13.8077234

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.7754974, 25.8881168, -7.7517147, 25.8817062, -30.5546722, 30.5391464
1: -0.1792345, 26.7474289, -0.1776915, 26.7392502, -21.1797600, 21.1861115
2: -0.2139604, 25.7863884, -0.1985724, 25.7632084, -20.6613388, 20.6731758
3: -4.5618372, 22.4709663, -4.5577354, 22.4319935, -19.7203560, 19.7477608
4: -7.4344215, 22.4770317, -7.4226689, 22.4622822, -25.3848724, 25.3882217
5: -4.6345124, 24.7991638, -4.6273193, 24.7699966, -23.5544853, 23.5765839
6: -39.3933144, -4.2942610, -39.3703957, -4.3386078, -29.5554047, 29.5873032
7: -8.9948511, 23.3653927, -8.9943886, 23.3822250, -25.5407486, 25.5385895
8: -13.6644096, 19.9503365, -13.6150913, 19.9150925, -29.0092163, 29.0053215
9: -8.4438286, 22.4680042, -8.4367523, 22.4640903, -27.6499863, 27.6159821
10: -28.8608780, 17.8001385, -28.8020592, 17.7746353, -39.4404907, 39.4054489
11: -26.1582832, 6.7330012, -26.1074333, 6.7322025, -29.5008774, 29.4469147
12: -46.0551147, -8.5641041, -46.0014915, -8.5993614, -32.0834045, 32.0584793
13: -32.4863701, 13.1795874, -32.5305672, 13.1649294, -38.3431778, 38.3946457
14: -59.3122978, -2.2459259, -59.2081642, -2.2715263, -56.5792236, 56.5018921
15: -14.0955124, 18.7686996, -14.0554991, 18.7476215, -27.8382492, 27.8135033
16: -15.5117302, 22.2187729, -15.5164528, 22.2513885, -29.4744797, 29.4490891
17: -59.0705681, -7.0230408, -59.0108414, -7.0108566, -51.1334381, 51.0495300
18: -21.9862461, 16.2245922, -21.9374790, 16.2299347, -32.9193192, 32.9095688
19: -22.0836868, 5.9220986, -22.0429440, 5.9231510, -21.7997513, 21.7558632
20: -27.7682343, 0.4822783, -27.7285194, 0.4636574, -22.1541634, 22.1296463
21: -26.2369938, 7.2476459, -26.1677132, 7.2226539, -28.6394730, 28.5890732
22: -29.3994827, 5.1787672, -29.4009743, 5.1759768, -27.3096085, 27.2676315
23: -17.7590733, 12.1730785, -17.7310028, 12.1916504, -23.8659821, 23.8287086
24: -16.4274921, 13.5528488, -16.4209595, 13.5647917, -25.6993561, 25.6933441
25: -23.7283154, 8.7482452, -23.7376213, 8.7559891, -27.7316589, 27.7339172
26: -39.2544479, 3.8468046, -39.2034569, 3.8563757, -32.9737244, 32.9142838
27: -19.4429893, 14.7946348, -19.4200592, 14.7936974, -34.2366867, 34.2146950
28: -21.2986183, 11.3098984, -21.2767086, 11.3178396, -26.0826111, 26.0541382
29: -24.3325481, 6.3073473, -24.3160286, 6.3063912, -28.4501114, 28.4019241
30: -29.9620991, 5.7390366, -29.9218292, 5.7261081, -33.7831192, 33.7628098
31: -23.2734756, 7.5419073, -23.2381039, 7.5455213, -24.8650513, 24.8317566
32: -36.9639435, -2.7202902, -36.9490700, -2.7606206, -29.6311111, 29.6514587
33: -54.3260651, -0.0776768, -54.3164902, -0.1478252, -44.1770935, 44.2350998
34: -49.0839691, -7.1207190, -49.0762787, -7.1508970, -35.6658936, 35.6715622
35: -40.2268333, 3.7768288, -40.2231750, 3.7379332, -35.8077545, 35.8361893
36: -45.5286865, 1.1733665, -45.5329437, 1.1426134, -39.4748154, 39.4909897
37: -60.6266937, -10.0056887, -60.6127853, -10.0090971, -44.7017670, 44.7102661
38: -53.5568314, 3.8641195, -53.5528412, 3.8198967, -48.5220642, 48.5837173
39: -61.8431549, -4.7037220, -61.8421173, -4.7393284, -48.0755463, 48.1094513
40: -50.1308212, -9.2259359, -50.1155548, -9.2280493, -38.9697723, 38.9703827
41: -31.9489670, 7.1875610, -31.9469147, 7.1788902, -37.8127213, 37.8196869
42: -30.0680695, -0.4071655, -30.0801277, -0.4099503, -22.9686890, 22.9695015

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7390582, upper bound: 13.6863122
time: 30.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7338570
time: 31.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.8135796, 25.9106770, -7.7573233, 25.8928490, -30.6035652, 30.5590897
1: -0.2113390, 26.7809601, -0.1802111, 26.7557049, -21.2304230, 21.1987762
2: -0.2390237, 25.8148766, -0.2016900, 25.7774487, -20.7024612, 20.6864586
3: -4.5887990, 22.5075588, -4.5597043, 22.4504166, -19.7673531, 19.7566032
4: -7.4719791, 22.5238647, -7.4266005, 22.4855137, -25.4497452, 25.4175987
5: -4.6651316, 24.8309212, -4.6324310, 24.7860775, -23.6041298, 23.5966225
6: -39.4038239, -4.2831430, -39.3728218, -4.3347158, -29.5724716, 29.5979462
7: -9.0294371, 23.3962059, -9.0000658, 23.3974228, -25.5920753, 25.5530205
8: -13.6933765, 19.9835129, -13.6167402, 19.9306316, -29.0516815, 29.0124016
9: -8.4624214, 22.4937382, -8.4393930, 22.4769554, -27.6842041, 27.6330795
10: -28.8791046, 17.8208237, -28.8046017, 17.7842598, -39.4742126, 39.4159775
11: -26.1904907, 6.7541361, -26.1203690, 6.7337060, -29.5210342, 29.4830666
12: -46.0690536, -8.5407944, -46.0075722, -8.5942430, -32.0981674, 32.0772247
13: -32.5123138, 13.1968269, -32.5424156, 13.1681433, -38.3628159, 38.4085312
14: -59.3377075, -2.2397404, -59.2182922, -2.2699385, -56.6002808, 56.5165405
15: -14.1223984, 18.8005276, -14.0585670, 18.7628136, -27.8825607, 27.8390884
16: -15.5328674, 22.2412453, -15.5202789, 22.2633629, -29.5108643, 29.4528999
17: -59.1206665, -6.9947462, -59.0307312, -7.0085039, -51.1565018, 51.0991211
18: -22.0080051, 16.2305565, -21.9424896, 16.2321014, -32.9426117, 32.9167099
19: -22.1289501, 5.9429817, -22.0645504, 5.9235840, -21.8124390, 21.7976036
20: -27.8077393, 0.4995489, -27.7478333, 0.4644384, -22.1750259, 22.1705437
21: -26.2751503, 7.2670631, -26.1853600, 7.2241411, -28.6508255, 28.6239624
22: -29.4340363, 5.1971979, -29.4170513, 5.1782055, -27.3161621, 27.2988205
23: -17.8017979, 12.2001982, -17.7516270, 12.1942682, -23.8917122, 23.8806076
24: -16.4644966, 13.5689182, -16.4389801, 13.5657139, -25.7244263, 25.7307663
25: -23.7991829, 8.7857094, -23.7727928, 8.7597647, -27.7688828, 27.8113480
26: -39.2853775, 3.8655815, -39.2179527, 3.8596411, -32.9909668, 32.9502411
27: -19.4520111, 14.7997913, -19.4240494, 14.7953167, -34.2473297, 34.2238388
28: -21.3450432, 11.3363361, -21.2995014, 11.3197861, -26.1157913, 26.1073990
29: -24.3574753, 6.3206491, -24.3266869, 6.3079562, -28.4626465, 28.4204979
30: -29.9953194, 5.7625098, -29.9376888, 5.7285757, -33.8123398, 33.8038406
31: -23.3179722, 7.5675559, -23.2589283, 7.5465565, -24.8799210, 24.8813515
32: -36.9736938, -2.7092452, -36.9518661, -2.7568111, -29.6463165, 29.6644745
33: -54.3515434, -0.0579529, -54.3288651, -0.1454744, -44.1818848, 44.2653961
34: -49.0933228, -7.1089592, -49.0780106, -7.1470022, -35.6763916, 35.6864319
35: -40.2430382, 3.7861233, -40.2309341, 3.7393303, -35.8088074, 35.8590164
36: -45.5595932, 1.1941805, -45.5483894, 1.1441936, -39.4817810, 39.5152130
37: -60.6695557, -9.9830513, -60.6333160, -10.0070248, -44.7131348, 44.7429276
38: -53.5854416, 3.8798561, -53.5660629, 3.8229408, -48.5367126, 48.6052704
39: -61.8840752, -4.6762524, -61.8623848, -4.7362127, -48.0868683, 48.1477280
40: -50.1452446, -9.2159157, -50.1186867, -9.2255278, -38.9898071, 38.9797211
41: -31.9630623, 7.2106881, -31.9497128, 7.1894326, -37.8375931, 37.8424530
42: -30.0869732, -0.3885341, -30.0892525, -0.4056463, -22.9882278, 22.9963341

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7072667
time: 32.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7549179
time: 29.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.8316498, 25.8889999, -7.8722916, 25.9016666, -30.6331940, 30.6386070
1: -0.2101941, 26.7485313, -0.2454863, 26.7642536, -21.2401772, 21.2528725
2: -0.2725587, 25.7883377, -0.3188872, 25.8037338, -20.7605133, 20.7632141
3: -4.6276631, 22.4730759, -4.6916289, 22.4824753, -19.8443832, 19.8208771
4: -7.4880495, 22.4799004, -7.5347586, 22.5049400, -25.4784660, 25.4581108
5: -4.7069564, 24.8030186, -4.7754059, 24.8202858, -23.6809158, 23.6671448
6: -39.3996582, -4.2856617, -39.3902626, -4.3159342, -29.5830231, 29.6202698
7: -9.0383177, 23.3677750, -9.0931988, 23.3978271, -25.6240005, 25.6463242
8: -13.7096481, 19.9543724, -13.7100048, 19.9594612, -29.1010208, 29.0973091
9: -8.4532213, 22.4891167, -8.4642105, 22.5113716, -27.7054825, 27.7019730
10: -28.8668213, 17.8530235, -28.8519058, 17.8882236, -39.5448685, 39.5055466
11: -26.1618443, 6.7764850, -26.1872139, 6.8202600, -29.4984894, 29.5817184
12: -46.0582237, -8.4505253, -46.0768814, -8.3673744, -32.2567978, 32.2462692
13: -32.4947205, 13.1946144, -32.5498009, 13.2052135, -38.3817596, 38.4798355
14: -59.3311996, -2.1503811, -59.3222237, -2.0779953, -56.7132721, 56.7105255
15: -14.1180592, 18.7710762, -14.1067219, 18.7698975, -27.8856201, 27.8423119
16: -15.5258751, 22.2280750, -15.5641594, 22.2730179, -29.4998550, 29.5082436
17: -59.0777092, -6.9689236, -59.1008911, -6.8980761, -51.2112274, 51.1903229
18: -21.9915237, 16.2585335, -22.0033817, 16.2956944, -32.9603271, 32.9584045
19: -22.0929661, 5.9564638, -22.1043949, 5.9918418, -21.8316422, 21.8573685
20: -27.7758102, 0.5291705, -27.7846413, 0.5586896, -22.1960373, 22.2451820
21: -26.2456264, 7.3027029, -26.2453651, 7.3336611, -28.6930008, 28.7343445
22: -29.4061623, 5.1956234, -29.4264183, 5.2169237, -27.3612976, 27.3774796
23: -17.7668190, 12.2058792, -17.7867470, 12.2589550, -23.8975067, 23.9022903
24: -16.4336090, 13.5614243, -16.4394722, 13.5832396, -25.7504196, 25.7103424
25: -23.7324257, 8.7721758, -23.7655373, 8.8060846, -27.7568893, 27.7765846
26: -39.2595139, 3.9417295, -39.2899628, 4.0443363, -33.0829315, 33.0952911
27: -19.4560699, 14.8003368, -19.4543438, 14.8097324, -34.2658005, 34.2546806
28: -21.3095703, 11.3468933, -21.3331299, 11.3930626, -26.1224976, 26.1443558
29: -24.3394718, 6.3268194, -24.3555603, 6.3502960, -28.5062408, 28.5091362
30: -29.9656754, 5.7694201, -29.9625225, 5.7919712, -33.8180084, 33.8276520
31: -23.2848701, 7.5670409, -23.2990284, 7.5965672, -24.8865967, 24.9049225
32: -36.9686699, -2.6900706, -36.9715004, -2.6960306, -29.6914749, 29.7100525
33: -54.3683395, -0.0661163, -54.4038315, -0.0819044, -44.2835388, 44.2715149
34: -49.1017532, -7.1149116, -49.1163445, -7.1143808, -35.7120667, 35.6995087
35: -40.2406998, 3.7787037, -40.2529449, 3.7571106, -35.8466339, 35.8541031
36: -45.5334473, 1.1859207, -45.5482597, 1.1769800, -39.5250854, 39.5515289
37: -60.6335297, -9.9820290, -60.6473236, -9.9576912, -44.7673340, 44.7608032
38: -53.5630760, 3.8812103, -53.5829887, 3.8633766, -48.5896912, 48.6164703
39: -61.8516846, -4.6890955, -61.8635025, -4.6924677, -48.1379242, 48.1512146
40: -50.1405602, -9.2207060, -50.1481705, -9.2071266, -39.0273895, 38.9996338
41: -31.9578667, 7.1969929, -31.9736786, 7.2015157, -37.8479004, 37.8565598
42: -30.0731068, -0.3933601, -30.0951939, -0.3767881, -22.9985123, 23.0047531

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7056513
time: 60.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7532015
time: 33.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.8697319, 25.9115982, -7.8778763, 25.9127998, -30.6821022, 30.6585083
1: -0.2422929, 26.7821045, -0.2480183, 26.7807350, -21.2908745, 21.2655678
2: -0.2976012, 25.8168488, -0.3219545, 25.8179607, -20.8016396, 20.7764740
3: -4.6546183, 22.5096302, -4.6935930, 22.5009003, -19.8913689, 19.8297272
4: -7.5256748, 22.5267525, -7.5387168, 22.5281487, -25.5433197, 25.4874268
5: -4.7375603, 24.8347759, -4.7805543, 24.8363304, -23.7305527, 23.6872025
6: -39.4100990, -4.2745829, -39.3926544, -4.3119850, -29.6001129, 29.6309433
7: -9.0728951, 23.3986206, -9.0989361, 23.4130211, -25.6753159, 25.6607590
8: -13.7385778, 19.9875374, -13.7116404, 19.9749756, -29.1435661, 29.1043930
9: -8.4717960, 22.5148354, -8.4668331, 22.5243034, -27.7396545, 27.7190628
10: -28.8850422, 17.8737335, -28.8545341, 17.8978920, -39.5785980, 39.5160141
11: -26.1941643, 6.7976284, -26.2001114, 6.8218083, -29.5186310, 29.6179199
12: -46.0721664, -8.4272003, -46.0829315, -8.3622837, -32.2715988, 32.2649765
13: -32.5206490, 13.2118311, -32.5616531, 13.2084312, -38.4013824, 38.4937363
14: -59.3565979, -2.1442013, -59.3324280, -2.0764675, -56.7343292, 56.7251129
15: -14.1449795, 18.8028984, -14.1097527, 18.7850800, -27.9299088, 27.8678665
16: -15.5469284, 22.2505798, -15.5679474, 22.2850075, -29.5362320, 29.5120659
17: -59.1278534, -6.9406567, -59.1207199, -6.8957567, -51.2343979, 51.2399750
18: -22.0132523, 16.2645092, -22.0083313, 16.2978420, -32.9836578, 32.9655762
19: -22.1382370, 5.9773493, -22.1259995, 5.9923344, -21.8443527, 21.8991318
20: -27.8153000, 0.5463715, -27.8039398, 0.5594783, -22.2169342, 22.2860909
21: -26.2838345, 7.3221135, -26.2629948, 7.3351460, -28.7043610, 28.7692375
22: -29.4407425, 5.2140269, -29.4424667, 5.2191172, -27.3678360, 27.4086571
23: -17.8095169, 12.2329674, -17.8073444, 12.2615929, -23.9232750, 23.9542007
24: -16.4706287, 13.5774717, -16.4574776, 13.5842037, -25.7755203, 25.7477646
25: -23.8032837, 8.8096724, -23.8006783, 8.8098545, -27.7941437, 27.8540192
26: -39.2904320, 3.9604597, -39.3044624, 4.0475159, -33.1001740, 33.1312027
27: -19.4651165, 14.8055496, -19.4583168, 14.8113194, -34.2764359, 34.2638664
28: -21.3559990, 11.3732281, -21.3559380, 11.3950415, -26.1557465, 26.1976242
29: -24.3643799, 6.3400669, -24.3661861, 6.3518591, -28.5187607, 28.5276833
30: -29.9988995, 5.7928796, -29.9783249, 5.7944140, -33.8472366, 33.8686676
31: -23.3293438, 7.5926800, -23.3198452, 7.5975862, -24.9015350, 24.9545326
32: -36.9784698, -2.6790285, -36.9742317, -2.6922235, -29.7066422, 29.7230301
33: -54.3938103, -0.0464067, -54.4162407, -0.0794783, -44.2882996, 44.3017273
34: -49.1111031, -7.1031246, -49.1180763, -7.1104465, -35.7225723, 35.7143860
35: -40.2568436, 3.7880077, -40.2607193, 3.7585506, -35.8476868, 35.8769379
36: -45.5643349, 1.2066870, -45.5637016, 1.1786480, -39.5320587, 39.5757828
37: -60.6764717, -9.9594231, -60.6678581, -9.9556217, -44.7787476, 44.7935791
38: -53.5917168, 3.8969412, -53.5960846, 3.8664207, -48.6044159, 48.6379776
39: -61.8926659, -4.6616659, -61.8837852, -4.6893644, -48.1493378, 48.1895065
40: -50.1549416, -9.2107077, -50.1512794, -9.2045164, -39.0474091, 39.0089569
41: -31.9719772, 7.2201009, -31.9764309, 7.2120833, -37.8727951, 37.8793640
42: -30.0920448, -0.3747778, -30.1042671, -0.3724899, -23.0180283, 23.0315628

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7266048
time: 29.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7742643
time: 34.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.9017248, 25.9114761, -7.8147354, 25.8836079, -30.6817703, 30.6264038
1: -0.2630215, 26.7663994, -0.2195950, 26.7400055, -21.2511444, 21.2478180
2: -0.2823229, 25.8077793, -0.2333829, 25.7642975, -20.7135162, 20.7290611
3: -4.6165357, 22.4851913, -4.5858049, 22.4340916, -19.7631950, 19.7938538
4: -7.5292430, 22.4920101, -7.4700665, 22.4636288, -25.4608002, 25.4529266
5: -4.7034931, 24.8216667, -4.6628237, 24.7725372, -23.6077385, 23.6364479
6: -39.4062233, -4.2233410, -39.3745003, -4.3037949, -29.6221161, 29.6638336
7: -9.0957460, 23.3956947, -9.0458956, 23.3838234, -25.6134605, 25.6219482
8: -13.7549820, 19.9729347, -13.6616735, 19.9180031, -29.0854378, 29.0753899
9: -8.5374365, 22.4991512, -8.4826612, 22.4655437, -27.7248840, 27.6940727
10: -28.9424515, 17.8325214, -28.8435993, 17.7793655, -39.5016861, 39.4775162
11: -26.1973877, 6.7521191, -26.1267853, 6.7409210, -29.5578041, 29.4858398
12: -46.0740547, -8.4809847, -46.0049438, -8.5583649, -32.1391678, 32.1529846
13: -32.5378609, 13.2024364, -32.5562134, 13.1729994, -38.4061356, 38.4735641
14: -59.3990021, -2.2155008, -59.2507210, -2.2677097, -56.6747894, 56.6027527
15: -14.1365566, 18.7868633, -14.0738754, 18.7576923, -27.8909683, 27.8568878
16: -15.6103077, 22.2594357, -15.5645714, 22.2524719, -29.5343781, 29.5393066
17: -59.1364059, -6.9995089, -59.0433960, -7.0023432, -51.2101593, 51.1672668
18: -22.0159969, 16.2616501, -21.9460430, 16.2483616, -32.9688339, 32.9516602
19: -22.1059456, 5.9561834, -22.0492630, 5.9405994, -21.8416901, 21.7914963
20: -27.7916222, 0.5272417, -27.7321129, 0.4861751, -22.2004623, 22.1713409
21: -26.2622223, 7.2811618, -26.1764889, 7.2395859, -28.6835556, 28.6290398
22: -29.4380875, 5.2348032, -29.4078026, 5.2050476, -27.3780670, 27.3016815
23: -17.7793159, 12.2141476, -17.7354259, 12.2118444, -23.9082184, 23.8616676
24: -16.4474144, 13.5868568, -16.4251862, 13.5818701, -25.7365112, 25.7231445
25: -23.7549019, 8.8008175, -23.7410183, 8.7828465, -27.7863159, 27.7649117
26: -39.2963028, 3.9235229, -39.2087288, 3.8952370, -33.0546417, 32.9724960
27: -19.4724579, 14.8348160, -19.4300308, 14.8137150, -34.2861710, 34.2648468
28: -21.3248787, 11.3649092, -21.2809982, 11.3461876, -26.1399384, 26.0989571
29: -24.3638973, 6.3510208, -24.3246117, 6.3289442, -28.5050049, 28.4363976
30: -29.9742393, 5.7609940, -29.9268322, 5.7360415, -33.8115311, 33.7937012
31: -23.2989712, 7.5849705, -23.2451668, 7.5670300, -24.9154205, 24.8828506
32: -36.9789658, -2.6674314, -36.9533844, -2.7342772, -29.6845856, 29.7107773
33: -54.3673210, -0.0023203, -54.3223839, -0.1101284, -44.2602997, 44.2777939
34: -49.1284447, -7.0300617, -49.0795784, -7.1057773, -35.7559662, 35.7453613
35: -40.2726250, 3.8616285, -40.2275696, 3.7807980, -35.8952637, 35.8935394
36: -45.5810204, 1.2709599, -45.5376854, 1.1917210, -39.5768433, 39.5777054
37: -60.6718140, -9.9347820, -60.6188469, -9.9726429, -44.7858124, 44.7485275
38: -53.6277618, 3.9949713, -53.5595512, 3.8847532, -48.6602325, 48.6989899
39: -61.8853531, -4.6430893, -61.8523903, -4.7091007, -48.1492004, 48.1692886
40: -50.1543427, -9.1938534, -50.1240959, -9.2124977, -39.0146790, 39.0140686
41: -31.9782372, 7.2511320, -31.9517460, 7.2105670, -37.8813629, 37.8880539
42: -30.0852470, -0.3475580, -30.0827427, -0.3801622, -23.0115738, 23.0295448

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7610865, upper bound: 13.6868713
time: 38.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7343751
time: 30.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9397678, 25.9340363, -7.8203688, 25.8947353, -30.7306519, 30.6463470
1: -0.2950735, 26.7999382, -0.2221107, 26.7564697, -21.3017921, 21.2605057
2: -0.3073878, 25.8362961, -0.2364614, 25.7785225, -20.7546501, 20.7423286
3: -4.6435046, 22.5217552, -4.5877948, 22.4525185, -19.8101768, 19.8027191
4: -7.5668421, 22.5388527, -7.4740195, 22.4868526, -25.5256958, 25.4822807
5: -4.7340918, 24.8534393, -4.6679316, 24.7886124, -23.6573868, 23.6564751
6: -39.4167175, -4.2123127, -39.3768921, -4.2999153, -29.6391830, 29.6744690
7: -9.1303768, 23.4265251, -9.0515833, 23.3990250, -25.6647758, 25.6363831
8: -13.7839794, 20.0061035, -13.6633101, 19.9335365, -29.1279945, 29.0824776
9: -8.5560312, 22.5248909, -8.4853649, 22.4784451, -27.7590790, 27.7111435
10: -28.9606705, 17.8531914, -28.8461285, 17.7889748, -39.5353470, 39.4879837
11: -26.2296009, 6.7732725, -26.1396770, 6.7424750, -29.5779076, 29.5220070
12: -46.0879936, -8.4576607, -46.0110245, -8.5532684, -32.1539001, 32.1716957
13: -32.5637131, 13.2196312, -32.5680695, 13.1761665, -38.4258194, 38.4874268
14: -59.4243088, -2.2093868, -59.2607956, -2.2661705, -56.6958466, 56.6173859
15: -14.1634541, 18.8186836, -14.0769062, 18.7728500, -27.9353104, 27.8824692
16: -15.6314278, 22.2819347, -15.5683823, 22.2644939, -29.5707779, 29.5431213
17: -59.1864929, -6.9712629, -59.0632935, -7.0000210, -51.2331390, 51.2168579
18: -22.0377560, 16.2676830, -21.9510384, 16.2505188, -32.9921494, 32.9588394
19: -22.1511745, 5.9770527, -22.0708828, 5.9409790, -21.8543854, 21.8332558
20: -27.8311348, 0.5445251, -27.7514648, 0.4870024, -22.2213593, 22.2122536
21: -26.3003674, 7.3005767, -26.1940479, 7.2411246, -28.6948700, 28.6639214
22: -29.4726372, 5.2532024, -29.4239044, 5.2071662, -27.3845825, 27.3328781
23: -17.8220711, 12.2412615, -17.7560425, 12.2144451, -23.9339600, 23.9135628
24: -16.4844513, 13.6029224, -16.4432106, 13.5828209, -25.7616119, 25.7605743
25: -23.8257694, 8.8382626, -23.7762070, 8.7866316, -27.8235016, 27.8423767
26: -39.3271790, 3.9422665, -39.2231598, 3.8984561, -33.0719452, 33.0084457
27: -19.4815140, 14.8400536, -19.4340172, 14.8153257, -34.2968407, 34.2740707
28: -21.3713112, 11.3912735, -21.3038368, 11.3481617, -26.1731415, 26.1522408
29: -24.3887615, 6.3643198, -24.3352776, 6.3305068, -28.5175171, 28.4549789
30: -30.0074806, 5.7844472, -29.9426136, 5.7384944, -33.8407288, 33.8347473
31: -23.3434830, 7.6105871, -23.2659760, 7.5680947, -24.9303131, 24.9324455
32: -36.9887390, -2.6564441, -36.9561577, -2.7304831, -29.6997452, 29.7237701
33: -54.3927536, 0.0174179, -54.3347549, -0.1076994, -44.2650452, 44.3080063
34: -49.1378098, -7.0182791, -49.0813141, -7.1018572, -35.7664566, 35.7602310
35: -40.2888641, 3.8708572, -40.2353172, 3.7822247, -35.8963394, 35.9163818
36: -45.6118736, 1.2918167, -45.5531311, 1.1933746, -39.5838013, 39.6019211
37: -60.7147598, -9.9121571, -60.6393738, -9.9706001, -44.7972260, 44.7812653
38: -53.6563835, 4.0107241, -53.5727501, 3.8877792, -48.6748505, 48.7205582
39: -61.9263000, -4.6155882, -61.8726730, -4.7059860, -48.1605225, 48.2076263
40: -50.1687889, -9.1838512, -50.1271858, -9.2099247, -39.0346680, 39.0234222
41: -31.9923439, 7.2741971, -31.9545021, 7.2211275, -37.9062500, 37.9108047
42: -30.1041603, -0.3289366, -30.0918388, -0.3758588, -23.0311203, 23.0563583

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7078267
time: 34.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7554483
time: 34.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.9578695, 25.9123745, -7.9353189, 25.9035492, -30.7603073, 30.7258263
1: -0.2939548, 26.7674961, -0.2873821, 26.7650223, -21.3115501, 21.3145752
2: -0.3409204, 25.8097534, -0.3536654, 25.8048210, -20.8127289, 20.8191032
3: -4.6824403, 22.4872952, -4.7197471, 22.4846039, -19.8872223, 19.8669891
4: -7.5829458, 22.4948730, -7.5821590, 22.5063038, -25.5543861, 25.5227661
5: -4.7759390, 24.8255310, -4.8109465, 24.8228493, -23.7342033, 23.7270355
6: -39.4125938, -4.2147865, -39.3943520, -4.2810774, -29.6497650, 29.6968155
7: -9.1392317, 23.3980923, -9.1447268, 23.3994045, -25.6967125, 25.7296677
8: -13.8002300, 19.9769516, -13.7565718, 19.9623756, -29.1772881, 29.1674004
9: -8.5468245, 22.5202255, -8.5101109, 22.5128670, -27.7803650, 27.7800293
10: -28.9484138, 17.8854027, -28.8934555, 17.8929920, -39.6060715, 39.5776062
11: -26.2009811, 6.7955947, -26.2065830, 6.8290739, -29.5553970, 29.6206627
12: -46.0771446, -8.3673458, -46.0803375, -8.3263569, -32.3125458, 32.3407784
13: -32.5462456, 13.2174187, -32.5754166, 13.2132874, -38.4446869, 38.5588150
14: -59.4179153, -2.1199694, -59.3647995, -2.0742817, -56.8087616, 56.8113861
15: -14.1591349, 18.7892647, -14.1251316, 18.7799301, -27.9383469, 27.8857002
16: -15.6243916, 22.2687149, -15.6122780, 22.2741585, -29.5597534, 29.5984573
17: -59.1435242, -6.9454479, -59.1334686, -6.8896303, -51.2880249, 51.3080139
18: -22.0212955, 16.2955704, -22.0119190, 16.3140373, -33.0098877, 33.0005798
19: -22.1152287, 5.9905272, -22.1106853, 6.0093164, -21.8735962, 21.8930588
20: -27.7991982, 0.5740910, -27.7882309, 0.5812259, -22.2423630, 22.2868881
21: -26.2708588, 7.3362632, -26.2541275, 7.3506246, -28.7370834, 28.7742996
22: -29.4447708, 5.2517099, -29.4332409, 5.2459679, -27.4297180, 27.4115105
23: -17.7870445, 12.2469225, -17.7911720, 12.2791538, -23.9397354, 23.9352531
24: -16.4535599, 13.5954800, -16.4437351, 13.6003637, -25.7875519, 25.7401352
25: -23.7589893, 8.8247719, -23.7689514, 8.8329601, -27.8115234, 27.8076057
26: -39.3013382, 4.0184116, -39.2952271, 4.0831461, -33.1639099, 33.1535110
27: -19.4855518, 14.8405714, -19.4643230, 14.8297386, -34.3152924, 34.3048935
28: -21.3358002, 11.4018688, -21.3374672, 11.4214172, -26.1798553, 26.1891632
29: -24.3708000, 6.3704634, -24.3641701, 6.3729081, -28.5611649, 28.5435829
30: -29.9778366, 5.7913861, -29.9674835, 5.8019724, -33.8464203, 33.8585587
31: -23.3102913, 7.6100492, -23.3061180, 7.6180725, -24.9369507, 24.9560280
32: -36.9837532, -2.6371951, -36.9757500, -2.6697040, -29.7448883, 29.7693329
33: -54.4095764, 0.0092087, -54.4097023, -0.0441837, -44.3666840, 44.3141403
34: -49.1462326, -7.0242691, -49.1196213, -7.0692978, -35.8020935, 35.7733307
35: -40.2864914, 3.8635588, -40.2573090, 3.8000002, -35.9341583, 35.9114685
36: -45.5857544, 1.2835236, -45.5529442, 1.2261362, -39.6271667, 39.6383362
37: -60.6786423, -9.9110947, -60.6534233, -9.9213066, -44.8513031, 44.7990952
38: -53.6340141, 4.0121374, -53.5896149, 3.9282036, -48.7278748, 48.7317352
39: -61.8938522, -4.6283951, -61.8737946, -4.6622438, -48.2115326, 48.2110367
40: -50.1640701, -9.1886873, -50.1566505, -9.1915464, -39.0722961, 39.0433197
41: -31.9871254, 7.2605557, -31.9784985, 7.2331815, -37.9165421, 37.9249191
42: -30.0903091, -0.3337379, -30.0977631, -0.3470297, -23.0413971, 23.0648155

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7062105
time: 30.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7537205
time: 29.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9959831, 25.9349670, -7.9409637, 25.9146976, -30.8091850, 30.7457695
1: -0.3260450, 26.8010540, -0.2898817, 26.7814865, -21.3622208, 21.3272781
2: -0.3659921, 25.8382683, -0.3567820, 25.8190479, -20.8538589, 20.8323936
3: -4.7093687, 22.5238094, -4.7217093, 22.5030136, -19.9342194, 19.8758316
4: -7.6204901, 22.5417252, -7.5861082, 22.5295143, -25.6192627, 25.5521202
5: -4.8065453, 24.8572998, -4.8160501, 24.8389206, -23.7838402, 23.7471237
6: -39.4230385, -4.2036829, -39.3967628, -4.2771711, -29.6668091, 29.7074661
7: -9.1738205, 23.4289036, -9.1504116, 23.4146042, -25.7480240, 25.7441330
8: -13.8291988, 20.0101357, -13.7582378, 19.9779053, -29.2198181, 29.1745110
9: -8.5654202, 22.5459595, -8.5127602, 22.5257530, -27.8145447, 27.7971039
10: -28.9666214, 17.9061012, -28.8959942, 17.9026527, -39.6397629, 39.5880585
11: -26.2332726, 6.8167415, -26.2194977, 6.8306031, -29.5755157, 29.6568832
12: -46.0910645, -8.3440170, -46.0863800, -8.3212414, -32.3273773, 32.3594627
13: -32.5721550, 13.2346458, -32.5872459, 13.2165174, -38.4643555, 38.5726471
14: -59.4433365, -2.1138077, -59.3749046, -2.0726452, -56.8299103, 56.8259735
15: -14.1860352, 18.8210773, -14.1281719, 18.7951145, -27.9826584, 27.9112625
16: -15.6454983, 22.2912674, -15.6160564, 22.2861328, -29.5961151, 29.6023216
17: -59.1936035, -6.9171877, -59.1533546, -6.8872795, -51.3109741, 51.3576508
18: -22.0430222, 16.3016205, -22.0168953, 16.3162384, -33.0332031, 33.0077515
19: -22.1604805, 6.0114374, -22.1323261, 6.0097413, -21.8862877, 21.9348145
20: -27.8387089, 0.5913744, -27.8075428, 0.5820084, -22.2632637, 22.3277588
21: -26.3090572, 7.3556757, -26.2717552, 7.3521099, -28.7484283, 28.8091888
22: -29.4793415, 5.2701120, -29.4493446, 5.2481318, -27.4362869, 27.4427223
23: -17.8297691, 12.2740412, -17.8117886, 12.2818184, -23.9654922, 23.9871559
24: -16.4905834, 13.6115417, -16.4617157, 13.6013355, -25.8126755, 25.7775803
25: -23.8298912, 8.8622465, -23.8041039, 8.8367310, -27.8487701, 27.8850937
26: -39.3322639, 4.0371423, -39.3097458, 4.0864077, -33.1811752, 33.1893997
27: -19.4945602, 14.8457794, -19.4683170, 14.8313637, -34.3259239, 34.3140945
28: -21.3822422, 11.4282475, -21.3602657, 11.4233789, -26.2130966, 26.2424545
29: -24.3957157, 6.3837481, -24.3747959, 6.3744640, -28.5736923, 28.5621452
30: -30.0110302, 5.8148270, -29.9832916, 5.8043957, -33.8756409, 33.8995972
31: -23.3548203, 7.6357188, -23.3269138, 7.6190772, -24.9518814, 25.0056534
32: -36.9934540, -2.6261473, -36.9785271, -2.6659222, -29.7600632, 29.7823639
33: -54.4350815, 0.0289164, -54.4221268, -0.0417347, -44.3715363, 44.3443146
34: -49.1556244, -7.0124416, -49.1213722, -7.0653067, -35.8126678, 35.7881470
35: -40.3027229, 3.8727789, -40.2650833, 3.8014393, -35.9351883, 35.9343262
36: -45.6166992, 1.3043137, -45.5684319, 1.2277851, -39.6341095, 39.6625595
37: -60.7216415, -9.8884602, -60.6739578, -9.9191399, -44.8627777, 44.8318100
38: -53.6626320, 4.0278568, -53.6028481, 3.9312754, -48.7425995, 48.7531967
39: -61.9348755, -4.6010847, -61.8940887, -4.6591616, -48.2229919, 48.2492676
40: -50.1784592, -9.1786451, -50.1597672, -9.1889744, -39.0923767, 39.0526505
41: -32.0012741, 7.2835908, -31.9812813, 7.2437725, -37.9414673, 37.9476700
42: -30.1092339, -0.3151612, -30.1068726, -0.3426995, -23.0609207, 23.0916138

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7271649
time: 32.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7271649
time: 155.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.8720617, 25.8911934, -7.9557362, 25.9067459, -30.6773300, 30.7438545
1: -0.2410612, 26.7494831, -0.3051825, 26.7632637, -21.2654648, 21.3007736
2: -0.2727680, 25.7885666, -0.3203614, 25.7878532, -20.7455902, 20.7747726
3: -4.6085491, 22.4772663, -4.6538115, 22.4601326, -19.7966194, 19.8344727
4: -7.5078087, 22.4801674, -7.5754976, 22.4823322, -25.4757957, 25.5176277
5: -4.6845655, 24.8024960, -4.7308092, 24.7942085, -23.6301117, 23.6641769
6: -39.3989983, -4.2222147, -39.4086342, -4.1854200, -29.7122574, 29.6977539
7: -9.0496817, 23.3673210, -9.1074543, 23.4011154, -25.6174660, 25.6331635
8: -13.7576876, 19.9550190, -13.8078871, 19.9578953, -29.1445694, 29.1655464
9: -8.5018625, 22.4726067, -8.5584574, 22.4927559, -27.7385254, 27.7274857
10: -28.9316177, 17.8094292, -28.9480591, 17.8251858, -39.5693817, 39.5362396
11: -26.1667099, 6.7717128, -26.1357002, 6.8134990, -29.5863571, 29.5152588
12: -46.0615005, -8.5169582, -46.0157471, -8.4974804, -32.2054443, 32.1237183
13: -32.5002136, 13.1987467, -32.5630913, 13.2110538, -38.4940109, 38.4532013
14: -59.3890533, -2.2376671, -59.3702354, -2.2214575, -56.7100677, 56.6724854
15: -14.1651154, 18.7767124, -14.2035789, 18.7852554, -27.9446564, 27.9647217
16: -15.5528488, 22.2204838, -15.6011791, 22.2717781, -29.5414963, 29.4936714
17: -59.1158180, -7.0130472, -59.1055489, -6.9715691, -51.2448196, 51.1528778
18: -22.0011482, 16.2550678, -21.9840984, 16.2931271, -33.0111084, 33.0242767
19: -22.0931454, 5.9685936, -22.0888672, 6.0188999, -21.8943596, 21.8444366
20: -27.7745132, 0.5367270, -27.7737026, 0.5760064, -22.2558823, 22.2279663
21: -26.2496147, 7.3100834, -26.2273235, 7.3517075, -28.7674332, 28.7096863
22: -29.4063683, 5.2218637, -29.4467220, 5.2630849, -27.3782272, 27.3621254
23: -17.7659111, 12.2077980, -17.7633018, 12.2642479, -23.9308777, 23.8946304
24: -16.4338760, 13.5915976, -16.4621658, 13.6442842, -25.7756042, 25.7726135
25: -23.7335091, 8.8029480, -23.7745056, 8.8708401, -27.8282089, 27.8355064
26: -39.2620163, 3.8905497, -39.2395020, 3.9439812, -33.0631332, 32.9991913
27: -19.4563465, 14.8401346, -19.4714508, 14.8881435, -34.3444901, 34.3115845
28: -21.3052311, 11.3645535, -21.3158436, 11.4308014, -26.1907806, 26.1475792
29: -24.3395882, 6.3390369, -24.3548431, 6.3727541, -28.5010910, 28.4767990
30: -29.9682350, 5.7964144, -29.9626141, 5.8469748, -33.9046326, 33.8617249
31: -23.2833595, 7.5952349, -23.2851048, 7.6563764, -24.9784546, 24.9287910
32: -36.9704590, -2.6718545, -36.9769478, -2.6581812, -29.7287598, 29.7314224
33: -54.3362389, -0.0011015, -54.3837624, 0.0089512, -44.2909698, 44.3831940
34: -49.0927391, -7.0653028, -49.1264877, -7.0376329, -35.7754898, 35.7778549
35: -40.2360992, 3.8463573, -40.2829628, 3.8795071, -35.9260406, 35.9684982
36: -45.5352783, 1.2514782, -45.5931091, 1.3023624, -39.5968399, 39.6294250
37: -60.6377602, -9.9561634, -60.6761284, -9.9075127, -44.7707825, 44.8244324
38: -53.5681114, 3.9616699, -53.6310463, 4.0207710, -48.6666260, 48.7557297
39: -61.8537903, -4.6371574, -61.9118500, -4.6030807, -48.1731415, 48.2500458
40: -50.1441345, -9.2104759, -50.1541748, -9.1949129, -39.0152283, 39.0435638
41: -31.9571228, 7.2337809, -31.9833565, 7.2758565, -37.9169235, 37.9049530
42: -30.0733376, -0.3741527, -30.0929241, -0.3393264, -23.0508728, 23.0241852

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7176638
time: 42.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7654948
time: 32.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.9101343, 25.9137669, -7.9613686, 25.9178848, -30.7262306, 30.7638168
1: -0.2732096, 26.7830200, -0.3076992, 26.7797241, -21.3161392, 21.3134499
2: -0.2978103, 25.8170528, -0.3234854, 25.8020668, -20.7867126, 20.7880363
3: -4.6355114, 22.5138817, -4.6557674, 22.4785194, -19.8436050, 19.8433151
4: -7.5453892, 22.5269985, -7.5794606, 22.5055466, -25.5406609, 25.5469513
5: -4.7151508, 24.8342686, -4.7359562, 24.8102913, -23.6797638, 23.6842194
6: -39.4094353, -4.2111406, -39.4110146, -4.1814642, -29.7293015, 29.7083664
7: -9.0843048, 23.3981361, -9.1131191, 23.4163036, -25.6688156, 25.6475601
8: -13.7866716, 19.9882126, -13.8095064, 19.9734497, -29.1871262, 29.1726265
9: -8.5204678, 22.4983234, -8.5611115, 22.5056915, -27.7727280, 27.7445488
10: -28.9498634, 17.8301086, -28.9506054, 17.8348732, -39.6031113, 39.5466843
11: -26.1990223, 6.7928567, -26.1486092, 6.8150463, -29.6064949, 29.5514565
12: -46.0754471, -8.4935818, -46.0218124, -8.4923382, -32.2200394, 32.1425285
13: -32.5260468, 13.2159052, -32.5749741, 13.2142773, -38.5137024, 38.4670868
14: -59.4144211, -2.2315083, -59.3803558, -2.2198772, -56.7311401, 56.6871643
15: -14.1920586, 18.8085232, -14.2066097, 18.8004494, -27.9889908, 27.9902802
16: -15.5739975, 22.2430038, -15.6050205, 22.2837715, -29.5779037, 29.4974747
17: -59.1658897, -6.9847736, -59.1254082, -6.9692793, -51.2678375, 51.2025299
18: -22.0228691, 16.2610741, -21.9890919, 16.2952747, -33.0344238, 33.0314407
19: -22.1383896, 5.9894781, -22.1104565, 6.0192957, -21.9070778, 21.8862076
20: -27.8140030, 0.5539374, -27.7929993, 0.5767865, -22.2767563, 22.2688751
21: -26.2878094, 7.3295364, -26.2449493, 7.3532000, -28.7787323, 28.7446175
22: -29.4409199, 5.2402687, -29.4627876, 5.2652330, -27.3847961, 27.3932800
23: -17.8086357, 12.2349186, -17.7839146, 12.2668724, -23.9566040, 23.9464951
24: -16.4708881, 13.6076603, -16.4801979, 13.6452618, -25.8006973, 25.8100128
25: -23.8044128, 8.8404770, -23.8096809, 8.8746424, -27.8654327, 27.9129372
26: -39.2929535, 3.9092646, -39.2539902, 3.9471922, -33.0803680, 33.0351410
27: -19.4654217, 14.8453159, -19.4754467, 14.8897314, -34.3551521, 34.3207626
28: -21.3516693, 11.3909693, -21.3386841, 11.4328213, -26.2240143, 26.2008286
29: -24.3644753, 6.3522978, -24.3655281, 6.3743148, -28.5136032, 28.4953690
30: -30.0014458, 5.8199043, -29.9784489, 5.8493838, -33.9338837, 33.9027405
31: -23.3278332, 7.6208911, -23.3058891, 7.6574197, -24.9934082, 24.9784546
32: -36.9802246, -2.6608505, -36.9796829, -2.6544209, -29.7438736, 29.7443924
33: -54.3617249, 0.0185394, -54.3962021, 0.0113554, -44.2957611, 44.4134598
34: -49.1021652, -7.0535450, -49.1282578, -7.0337129, -35.7860260, 35.7927170
35: -40.2523384, 3.8555565, -40.2907104, 3.8810062, -35.9270706, 35.9913788
36: -45.5662079, 1.2722855, -45.6085396, 1.3039207, -39.6037903, 39.6536942
37: -60.6807098, -9.9335489, -60.6966743, -9.9054585, -44.7821808, 44.8570938
38: -53.5967751, 3.9774141, -53.6442833, 4.0237780, -48.6812744, 48.7773666
39: -61.8947678, -4.6098051, -61.9321213, -4.5999718, -48.1844025, 48.2883301
40: -50.1585007, -9.2004890, -50.1572914, -9.1923180, -39.0352478, 39.0529480
41: -31.9712372, 7.2568188, -31.9861736, 7.2863579, -37.9417572, 37.9277267
42: -30.0922585, -0.3555369, -30.1020164, -0.3350167, -23.0704155, 23.0509949

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7386184
time: 34.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7865368
time: 28.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.9281788, 25.8920898, -8.0763378, 25.9267235, -30.7558746, 30.8432922
1: -0.2720990, 26.7506046, -0.3729763, 26.7882690, -21.3258858, 21.3675346
2: -0.3313866, 25.7905140, -0.4406481, 25.8283691, -20.8447876, 20.8648376
3: -4.6744342, 22.4794102, -4.7877221, 22.5106258, -19.9206390, 19.9076042
4: -7.5615072, 22.4830151, -7.6875896, 22.5249710, -25.5693436, 25.5874710
5: -4.7569942, 24.8063583, -4.8789635, 24.8445435, -23.7565842, 23.7548141
6: -39.4053230, -4.2136059, -39.4284935, -4.1627026, -29.7398834, 29.7306900
7: -9.0931520, 23.3696766, -9.2063046, 23.4167080, -25.7007446, 25.7409134
8: -13.8029385, 19.9590435, -13.9027681, 20.0022411, -29.2364120, 29.2575645
9: -8.5112648, 22.4936943, -8.5858736, 22.5400658, -27.7939987, 27.8133774
10: -28.9375477, 17.8623142, -28.9978752, 17.9388428, -39.6737671, 39.6362686
11: -26.1703529, 6.8152280, -26.2154922, 6.9016361, -29.5839920, 29.6501389
12: -46.0645981, -8.4033356, -46.0911179, -8.2654581, -32.3788681, 32.3116074
13: -32.5085144, 13.2136822, -32.5823555, 13.2513247, -38.5326233, 38.5384140
14: -59.4079857, -2.1420918, -59.4842911, -2.0280495, -56.8441772, 56.8810272
15: -14.1877127, 18.7790222, -14.2548351, 18.8074951, -27.9919968, 27.9934769
16: -15.5669365, 22.2298145, -15.6488743, 22.2934589, -29.5668793, 29.5528107
17: -59.1229248, -6.9589596, -59.1955719, -6.8588581, -51.3226624, 51.2936707
18: -22.0063820, 16.2890091, -22.0499516, 16.3588715, -33.0521164, 33.0731812
19: -22.1024323, 6.0029736, -22.1503048, 6.0876207, -21.9262772, 21.9459915
20: -27.7821064, 0.5835428, -27.8298073, 0.6710567, -22.2977867, 22.3434792
21: -26.2582703, 7.3651571, -26.3050041, 7.4627395, -28.8209305, 28.8549500
22: -29.4130287, 5.2386971, -29.4721565, 5.3040457, -27.4298859, 27.4719658
23: -17.7736664, 12.2405930, -17.8190727, 12.3315821, -23.9624023, 23.9681969
24: -16.4399834, 13.6001825, -16.4807186, 13.6627989, -25.8266602, 25.7896118
25: -23.7376442, 8.8269253, -23.8024292, 8.9209347, -27.8535004, 27.8782043
26: -39.2670593, 3.9854302, -39.3260651, 4.1319351, -33.1723938, 33.1802368
27: -19.4694672, 14.8458481, -19.5057125, 14.9041357, -34.3736038, 34.3515625
28: -21.3161793, 11.4015427, -21.3723145, 11.5060539, -26.2307587, 26.2377968
29: -24.3464928, 6.3584557, -24.3944130, 6.4167242, -28.5572510, 28.5839577
30: -29.9718208, 5.8268280, -30.0033264, 5.9128346, -33.9395447, 33.9265976
31: -23.2946396, 7.6203842, -23.3460445, 7.7073832, -25.0000229, 25.0020027
32: -36.9751816, -2.6416397, -36.9993210, -2.5936213, -29.7891083, 29.7899551
33: -54.3785324, 0.0103798, -54.4711342, 0.0749483, -44.3973999, 44.4196014
34: -49.1105652, -7.0595188, -49.1665878, -7.0011444, -35.8216400, 35.8058167
35: -40.2499352, 3.8482304, -40.3127289, 3.8987207, -35.9649124, 35.9865112
36: -45.5401382, 1.2639837, -45.6084366, 1.3367424, -39.6471405, 39.6900330
37: -60.6445694, -9.9325104, -60.7107582, -9.8560848, -44.8363190, 44.8750687
38: -53.5744019, 3.9788351, -53.6611328, 4.0642633, -48.7343140, 48.7884445
39: -61.8623543, -4.6226130, -61.9332428, -4.5562325, -48.2354736, 48.2917938
40: -50.1538162, -9.2052326, -50.1868057, -9.1739197, -39.0729065, 39.0728073
41: -31.9660034, 7.2431598, -32.0101318, 7.2984166, -37.9521255, 37.9418869
42: -30.0783730, -0.3603239, -30.1079388, -0.3061647, -23.0807114, 23.0594292

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7369992
time: 34.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7848401
time: 37.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.9663057, 25.9146843, -8.0819683, 25.9378738, -30.8047333, 30.8632812
1: -0.3041167, 26.7841434, -0.3754940, 26.8047562, -21.3765640, 21.3801956
2: -0.3564448, 25.8190384, -0.4437618, 25.8426094, -20.8859100, 20.8781319
3: -4.7013173, 22.5159607, -4.7897043, 22.5290260, -19.9676208, 19.9164429
4: -7.5990968, 22.5298691, -7.6915193, 22.5482101, -25.6342354, 25.6168022
5: -4.7876549, 24.8381424, -4.8840914, 24.8605976, -23.8062134, 23.7748642
6: -39.4157639, -4.2025118, -39.4308777, -4.1587734, -29.7569351, 29.7413330
7: -9.1277704, 23.4005165, -9.2119904, 23.4319267, -25.7520447, 25.7553482
8: -13.8318834, 19.9922543, -13.9044266, 20.0177650, -29.2789650, 29.2646523
9: -8.5298271, 22.5194340, -8.5885763, 22.5529995, -27.8282166, 27.8304901
10: -28.9557915, 17.8830185, -29.0004234, 17.9484940, -39.7075119, 39.6467285
11: -26.2026482, 6.8363619, -26.2284317, 6.9031539, -29.6040840, 29.6863060
12: -46.0785255, -8.3800020, -46.0972023, -8.2603254, -32.3936386, 32.3302689
13: -32.5344048, 13.2309313, -32.5941811, 13.2545366, -38.5522766, 38.5522385
14: -59.4333687, -2.1359749, -59.4944534, -2.0264187, -56.8651276, 56.8956604
15: -14.2146397, 18.8108673, -14.2578754, 18.8227139, -28.0363312, 28.0190735
16: -15.5880489, 22.2523212, -15.6526566, 22.3054447, -29.6032562, 29.5566177
17: -59.1730499, -6.9307079, -59.2154236, -6.8565416, -51.3457336, 51.3433228
18: -22.0281239, 16.2950058, -22.0549507, 16.3610497, -33.0754089, 33.0803375
19: -22.1476974, 6.0238581, -22.1719532, 6.0880413, -21.9390182, 21.9877548
20: -27.8215847, 0.6008120, -27.8491573, 0.6718621, -22.3186569, 22.3843880
21: -26.2964573, 7.3846045, -26.3225994, 7.4642630, -28.8322906, 28.8898735
22: -29.4476585, 5.2571173, -29.4882317, 5.3061786, -27.4364395, 27.5031929
23: -17.8163986, 12.2677174, -17.8396587, 12.3342228, -23.9881363, 24.0200996
24: -16.4770203, 13.6162653, -16.4987335, 13.6637564, -25.8517609, 25.8270416
25: -23.8085079, 8.8643742, -23.8375969, 8.9247675, -27.8907013, 27.9556618
26: -39.2979507, 4.0041370, -39.3404922, 4.1352115, -33.1896362, 33.2161179
27: -19.4785213, 14.8510227, -19.5097122, 14.9057369, -34.3842583, 34.3607330
28: -21.3626518, 11.4278965, -21.3950768, 11.5080414, -26.2639923, 26.2910461
29: -24.3714218, 6.3717194, -24.4050446, 6.4182634, -28.5697403, 28.6025200
30: -30.0049992, 5.8502679, -30.0191517, 5.9153237, -33.9687881, 33.9675980
31: -23.3391418, 7.6460094, -23.3668671, 7.7083855, -25.0149765, 25.0516052
32: -36.9849243, -2.6306152, -37.0020638, -2.5897751, -29.8042526, 29.8029861
33: -54.4040146, 0.0300655, -54.4835510, 0.0772820, -44.4021454, 44.4497910
34: -49.1199417, -7.0476594, -49.1683273, -6.9972391, -35.8321762, 35.8206635
35: -40.2661743, 3.8574810, -40.3204956, 3.9001331, -35.9659424, 36.0093918
36: -45.5709763, 1.2848034, -45.6238708, 1.3383636, -39.6541519, 39.7143173
37: -60.6875305, -9.9099026, -60.7312775, -9.8540115, -44.8477325, 44.9077454
38: -53.6030579, 3.9944916, -53.6743393, 4.0673065, -48.7489777, 48.8100586
39: -61.9033432, -4.5951538, -61.9535370, -4.5531731, -48.2468719, 48.3300934
40: -50.1682129, -9.1952391, -50.1898727, -9.1713476, -39.0929260, 39.0821915
41: -31.9801006, 7.2662630, -32.0129089, 7.3089843, -37.9770508, 37.9646454
42: -30.0972977, -0.3417673, -30.1170120, -0.3018775, -23.1002312, 23.0862312

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7579500
time: 29.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7485743, upper bound: 13.8058847
time: 39.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.9982958, 25.9145393, -8.0188036, 25.9086342, -30.8044395, 30.8311310
1: -0.3248930, 26.7684498, -0.3470888, 26.7640266, -21.3368568, 21.3625031
2: -0.3411913, 25.8099709, -0.3551435, 25.7889404, -20.7977943, 20.8306885
3: -4.6632895, 22.4914818, -4.6818833, 22.4622192, -19.8394394, 19.8805809
4: -7.6027098, 22.4951363, -7.6229377, 22.4837189, -25.5517960, 25.5823479
5: -4.7535019, 24.8250008, -4.7663660, 24.7967548, -23.6833878, 23.7240868
6: -39.4119034, -4.1513004, -39.4127426, -4.1505165, -29.7790070, 29.7742920
7: -9.1506290, 23.3975983, -9.1589947, 23.4026928, -25.6902237, 25.7165375
8: -13.8483067, 19.9776306, -13.8544607, 19.9608135, -29.2208748, 29.2357101
9: -8.5955391, 22.5036888, -8.6044617, 22.4942322, -27.8134232, 27.8055763
10: -29.0131645, 17.8417740, -28.9895992, 17.8298759, -39.6305771, 39.6083374
11: -26.2058144, 6.7908764, -26.1550446, 6.8223047, -29.6432953, 29.5542068
12: -46.0804291, -8.4337387, -46.0192337, -8.4564095, -32.2612228, 32.2183266
13: -32.5516472, 13.2214613, -32.5887527, 13.2190495, -38.5569458, 38.5321579
14: -59.4758148, -2.2072783, -59.4127235, -2.2177019, -56.8056030, 56.7734528
15: -14.2063198, 18.7948170, -14.2220764, 18.7953110, -27.9974747, 28.0081329
16: -15.6513844, 22.2611618, -15.6493187, 22.2729111, -29.6013794, 29.5838852
17: -59.1817551, -6.9895716, -59.1381836, -6.9630671, -51.3215485, 51.2705994
18: -22.0308628, 16.2921295, -21.9926071, 16.3115253, -33.0606613, 33.0664520
19: -22.1153526, 6.0026746, -22.0951424, 6.0363035, -21.9363098, 21.8801155
20: -27.7979202, 0.5816493, -27.7772655, 0.5985599, -22.3022079, 22.2696724
21: -26.2748165, 7.3436604, -26.2360401, 7.3686590, -28.8114853, 28.7496567
22: -29.4449444, 5.2778649, -29.4535522, 5.2920728, -27.4466400, 27.3961830
23: -17.7861385, 12.2488842, -17.7677250, 12.2844753, -23.9731140, 23.9275742
24: -16.4538097, 13.6256361, -16.4664192, 13.6614628, -25.8127975, 25.8024368
25: -23.7601395, 8.8555489, -23.7778969, 8.8977356, -27.8828583, 27.8665314
26: -39.3038254, 3.9673042, -39.2447433, 3.9828286, -33.1440506, 33.0574341
27: -19.4858322, 14.8803978, -19.4813957, 14.9081335, -34.3939667, 34.3617935
28: -21.3314724, 11.4195757, -21.3201675, 11.4592171, -26.2481232, 26.1923943
29: -24.3708763, 6.3826933, -24.3634567, 6.3953481, -28.5559769, 28.5112762
30: -29.9803925, 5.8184180, -29.9675980, 5.8569722, -33.9331055, 33.8926239
31: -23.3087540, 7.6383305, -23.2921791, 7.6779180, -25.0288773, 24.9799309
32: -36.9854622, -2.6189275, -36.9812050, -2.6318483, -29.7822037, 29.7907562
33: -54.3774643, 0.0742397, -54.3896484, 0.0467796, -44.3741608, 44.4258499
34: -49.1372337, -6.9746480, -49.1297646, -6.9924693, -35.8655396, 35.8516235
35: -40.2819176, 3.9311342, -40.2873497, 3.9224501, -36.0135651, 36.0258942
36: -45.5875931, 1.3491154, -45.5977936, 1.3515234, -39.6988831, 39.7162094
37: -60.6828842, -9.8852215, -60.6821404, -9.8710327, -44.8547668, 44.8627014
38: -53.6389847, 4.0925465, -53.6377716, 4.0857000, -48.8047943, 48.8709717
39: -61.8959808, -4.5765190, -61.9221191, -4.5728779, -48.2467804, 48.3098373
40: -50.1676483, -9.1783648, -50.1626740, -9.1793509, -39.0601959, 39.0873108
41: -31.9864006, 7.2972913, -31.9881687, 7.3075180, -37.9856567, 37.9733429
42: -30.0905113, -0.3145227, -30.0954933, -0.3094754, -23.0937691, 23.0842323

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7182276
time: 34.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7660527
time: 31.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.0363617, 25.9371128, -8.0244484, 25.9197960, -30.8533287, 30.8510742
1: -0.3569460, 26.8019676, -0.3496118, 26.7804890, -21.3875504, 21.3751678
2: -0.3662214, 25.8384705, -0.3582492, 25.8031921, -20.8389206, 20.8439636
3: -4.6902728, 22.5280495, -4.6838775, 22.4806309, -19.8864594, 19.8894272
4: -7.6402988, 22.5419865, -7.6269159, 22.5069103, -25.6166840, 25.6117020
5: -4.7841406, 24.8567924, -4.7714500, 24.8128109, -23.7330170, 23.7441063
6: -39.4223480, -4.1402674, -39.4151001, -4.1466084, -29.7960358, 29.7849045
7: -9.1852026, 23.4284172, -9.1646729, 23.4179096, -25.7415161, 25.7309761
8: -13.8772545, 20.0108185, -13.8561182, 19.9763565, -29.2634315, 29.2427711
9: -8.6140690, 22.5294666, -8.6070805, 22.5071316, -27.8476105, 27.8226585
10: -29.0314369, 17.8625069, -28.9921074, 17.8395996, -39.6642838, 39.6187592
11: -26.2380848, 6.8120265, -26.1679688, 6.8238821, -29.6634216, 29.5904160
12: -46.0943909, -8.4104061, -46.0253067, -8.4512825, -32.2758789, 32.2371101
13: -32.5775604, 13.2387238, -32.6005936, 13.2222881, -38.5766602, 38.5459900
14: -59.5011444, -2.2011147, -59.4228516, -2.2160892, -56.8266907, 56.7880249
15: -14.2332382, 18.8266296, -14.2251301, 18.8104782, -28.0417938, 28.0337448
16: -15.6725082, 22.2836971, -15.6531067, 22.2849140, -29.6378021, 29.5877609
17: -59.2317123, -6.9613199, -59.1580658, -6.9608154, -51.3445282, 51.3202209
18: -22.0526085, 16.2981739, -21.9975834, 16.3136959, -33.0839844, 33.0735855
19: -22.1606445, 6.0235815, -22.1167736, 6.0367527, -21.9490356, 21.9218750
20: -27.8374081, 0.5989485, -27.7966022, 0.5993319, -22.3230743, 22.3105698
21: -26.3130093, 7.3630719, -26.2536716, 7.3701887, -28.8228455, 28.7845840
22: -29.4795456, 5.2962976, -29.4696274, 5.2942672, -27.4532089, 27.4273453
23: -17.8288803, 12.2759867, -17.7883434, 12.2870808, -23.9988327, 23.9794731
24: -16.4908447, 13.6417198, -16.4844608, 13.6623783, -25.8378677, 25.8398743
25: -23.8310032, 8.8930559, -23.8131046, 8.9015388, -27.9200439, 27.9439774
26: -39.3347397, 3.9859550, -39.2592239, 3.9860563, -33.1613464, 33.0933838
27: -19.4948578, 14.8855543, -19.4854050, 14.9097443, -34.4046021, 34.3709602
28: -21.3778877, 11.4459581, -21.3429737, 11.4611893, -26.2813950, 26.2456779
29: -24.3957882, 6.3959522, -24.3741531, 6.3968935, -28.5685196, 28.5298691
30: -30.0135994, 5.8418417, -29.9834213, 5.8593922, -33.9623032, 33.9336700
31: -23.3532791, 7.6639276, -23.3129654, 7.6789513, -25.0437775, 25.0295258
32: -36.9952316, -2.6079502, -36.9839706, -2.6280546, -29.7973709, 29.8037415
33: -54.4029427, 0.0938835, -54.4020691, 0.0491953, -44.3789673, 44.4561462
34: -49.1466255, -6.9628544, -49.1315460, -6.9885783, -35.8760529, 35.8665161
35: -40.2981606, 3.9403954, -40.2950783, 3.9238949, -36.0146027, 36.0487442
36: -45.6185570, 1.3698769, -45.6133041, 1.3530703, -39.7058105, 39.7404633
37: -60.7258606, -9.8626060, -60.7026978, -9.8689976, -44.8662109, 44.8953781
38: -53.6676292, 4.1082602, -53.6508980, 4.0887241, -48.8194275, 48.8925629
39: -61.9369812, -4.5491228, -61.9423981, -4.5697279, -48.2581024, 48.3481979
40: -50.1820488, -9.1683722, -50.1657562, -9.1767282, -39.0802002, 39.0966492
41: -32.0004921, 7.3203773, -31.9909668, 7.3179922, -38.0104828, 37.9960403
42: -30.1094170, -0.2959261, -30.1045685, -0.3052044, -23.1133041, 23.1110573

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7391863
time: 40.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7871037
time: 32.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0544462, 25.9154510, -8.1394234, 25.9286137, -30.8829803, 30.9306030
1: -0.3558340, 26.7695599, -0.4148540, 26.7890358, -21.3972626, 21.4292412
2: -0.3997822, 25.8119431, -0.4754653, 25.8294258, -20.8969803, 20.9207497
3: -4.7291722, 22.4935989, -4.8158841, 22.5127220, -19.9635048, 19.9537048
4: -7.6563478, 22.4979858, -7.7350092, 22.5263500, -25.6453323, 25.6522217
5: -4.8259640, 24.8288708, -4.9145083, 24.8470917, -23.8098869, 23.8147125
6: -39.4182281, -4.1426764, -39.4326096, -4.1278324, -29.8066406, 29.8072815
7: -9.1940937, 23.3999710, -9.2578239, 23.4183197, -25.7734756, 25.8242950
8: -13.8935452, 19.9816360, -13.9493389, 20.0051594, -29.3127136, 29.3276901
9: -8.6048660, 22.5248013, -8.6318579, 22.5415249, -27.8688583, 27.8914642
10: -29.0191689, 17.8946724, -29.0394402, 17.9435406, -39.7349548, 39.7083435
11: -26.2094727, 6.8343773, -26.2348442, 6.9104643, -29.6409073, 29.6890831
12: -46.0835266, -8.3201122, -46.0945892, -8.2243748, -32.4346771, 32.4061813
13: -32.5599480, 13.2364874, -32.6079750, 13.2593832, -38.5955505, 38.6172791
14: -59.4946899, -2.1117506, -59.5268250, -2.0242710, -56.9396057, 56.9819489
15: -14.2289104, 18.7971802, -14.2733393, 18.8175392, -28.0447693, 28.0369339
16: -15.6654987, 22.2705040, -15.6969681, 22.2945499, -29.6267624, 29.6430511
17: -59.1887283, -6.9354868, -59.2281990, -6.8503819, -51.3993835, 51.4114075
18: -22.0361176, 16.3260746, -22.0584755, 16.3772926, -33.1016617, 33.1153259
19: -22.1246681, 6.0370822, -22.1566238, 6.1050711, -21.9682388, 21.9816818
20: -27.8054714, 0.6285191, -27.8334198, 0.6936517, -22.3440857, 22.3851929
21: -26.2834587, 7.3987093, -26.3137512, 7.4796934, -28.8650436, 28.8949356
22: -29.4516201, 5.2947607, -29.4790001, 5.3330536, -27.4983444, 27.5060272
23: -17.7939014, 12.2816582, -17.8234844, 12.3518219, -24.0046539, 24.0011902
24: -16.4599419, 13.6342564, -16.4849510, 13.6799555, -25.8637848, 25.8194351
25: -23.7642326, 8.8795128, -23.8058281, 8.9478521, -27.9081039, 27.9092674
26: -39.3089409, 4.0621548, -39.3313293, 4.1707916, -33.2533417, 33.2384567
27: -19.4989223, 14.8860989, -19.5156765, 14.9242077, -34.4231300, 34.4017754
28: -21.3424091, 11.4565544, -21.3765945, 11.5344467, -26.2880707, 26.2826004
29: -24.3778114, 6.4021015, -24.4029865, 6.4392662, -28.6121368, 28.6184578
30: -29.9839859, 5.8487887, -30.0082703, 5.9228654, -33.9679871, 33.9574890
31: -23.3200970, 7.6634326, -23.3530846, 7.7289462, -25.0504150, 25.0531540
32: -36.9901848, -2.5886979, -37.0035782, -2.5672708, -29.8425293, 29.8493423
33: -54.4197502, 0.0857439, -54.4769516, 0.1127377, -44.4806213, 44.4621582
34: -49.1550522, -6.9687543, -49.1698380, -6.9560380, -35.9116821, 35.8796082
35: -40.2957916, 3.9330606, -40.3170967, 3.9416122, -36.0524368, 36.0438614
36: -45.5923615, 1.3616590, -45.6131439, 1.3858719, -39.7491913, 39.7768021
37: -60.6896973, -9.8615913, -60.7168198, -9.8195887, -44.9203186, 44.9133530
38: -53.6452103, 4.1096640, -53.6678543, 4.1291733, -48.8724365, 48.9037170
39: -61.9045525, -4.5618954, -61.9434814, -4.5259762, -48.3091125, 48.3516083
40: -50.1773224, -9.1731424, -50.1953011, -9.1583958, -39.1178436, 39.1165695
41: -31.9953117, 7.3067389, -32.0149536, 7.3301582, -38.0208282, 38.0102997
42: -30.0955334, -0.3006983, -30.1105003, -0.2763700, -23.1236076, 23.1195107

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7375659
time: 29.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7853966
time: 32.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.0925388, 25.9380302, -8.1450338, 25.9397602, -30.9318657, 30.9505577
1: -0.3878756, 26.8030930, -0.4173746, 26.8055077, -21.4479446, 21.4419022
2: -0.4248409, 25.8403969, -0.4785466, 25.8436661, -20.9381027, 20.9340363
3: -4.7561107, 22.5301285, -4.8178339, 22.5311508, -20.0104942, 19.9625702
4: -7.6939259, 22.5448494, -7.7389894, 22.5495834, -25.7102051, 25.6815414
5: -4.8565655, 24.8606339, -4.9195995, 24.8631420, -23.8595085, 23.8347740
6: -39.4286766, -4.1316462, -39.4349518, -4.1239166, -29.8236618, 29.8178940
7: -9.2286654, 23.4308224, -9.2635202, 23.4334984, -25.8247910, 25.8387222
8: -13.9224977, 20.0148430, -13.9510260, 20.0206909, -29.3552704, 29.3347855
9: -8.6234856, 22.5505486, -8.6345377, 22.5544491, -27.9030609, 27.9085617
10: -29.0373993, 17.9153938, -29.0419407, 17.9532394, -39.7687073, 39.7187958
11: -26.2417431, 6.8555422, -26.2477608, 6.9119830, -29.6610260, 29.7252731
12: -46.0974846, -8.2967758, -46.1006584, -8.2192650, -32.4494476, 32.4248886
13: -32.5858498, 13.2536554, -32.6197891, 13.2626200, -38.6152649, 38.6312103
14: -59.5200844, -2.1055584, -59.5369568, -2.0226440, -56.9606934, 56.9965820
15: -14.2558479, 18.8290195, -14.2763557, 18.8327293, -28.0891418, 28.0625076
16: -15.6866264, 22.2929974, -15.7007713, 22.3065395, -29.6631317, 29.6468887
17: -59.2388191, -6.9072294, -59.2480698, -6.8480415, -51.4223480, 51.4610291
18: -22.0578384, 16.3321476, -22.0634708, 16.3794632, -33.1249847, 33.1225052
19: -22.1699619, 6.0579143, -22.1782608, 6.1054702, -21.9809761, 22.0234528
20: -27.8449516, 0.6457896, -27.8527374, 0.6944103, -22.3649712, 22.4261055
21: -26.3216896, 7.4181409, -26.3313408, 7.4811921, -28.8764038, 28.9298477
22: -29.4861908, 5.3131795, -29.4950657, 5.3352017, -27.5049057, 27.5372238
23: -17.8366089, 12.3087788, -17.8440971, 12.3543968, -24.0303955, 24.0530777
24: -16.4969711, 13.6503038, -16.5029716, 13.6808872, -25.8889008, 25.8568726
25: -23.8351440, 8.9170017, -23.8409691, 8.9516687, -27.9453506, 27.9867172
26: -39.3398285, 4.0808916, -39.3458023, 4.1740813, -33.2706223, 33.2743988
27: -19.5079422, 14.8912849, -19.5196838, 14.9257851, -34.4337273, 34.4109688
28: -21.3888512, 11.4829216, -21.3994102, 11.5364170, -26.3213120, 26.3358383
29: -24.4027119, 6.4153976, -24.4136543, 6.4408183, -28.6246490, 28.6370354
30: -30.0171986, 5.8722420, -30.0241051, 5.9253187, -33.9972076, 33.9985123
31: -23.3646393, 7.6890574, -23.3739185, 7.7299838, -25.0653458, 25.1027374
32: -36.9999428, -2.5776944, -37.0063133, -2.5634871, -29.8577118, 29.8622894
33: -54.4452324, 0.1054325, -54.4893913, 0.1151590, -44.4853821, 44.4924088
34: -49.1643906, -6.9570580, -49.1716232, -6.9520645, -35.9222107, 35.8944244
35: -40.3119850, 3.9422836, -40.3248749, 3.9430809, -36.0534134, 36.0667343
36: -45.6232948, 1.3824186, -45.6285782, 1.3875027, -39.7561188, 39.8011169
37: -60.7327423, -9.8389664, -60.7373199, -9.8175793, -44.9317627, 44.9460678
38: -53.6739120, 4.1254158, -53.6809769, 4.1321630, -48.8870850, 48.9252777
39: -61.9455185, -4.5344849, -61.9638290, -4.5229492, -48.3205261, 48.3899384
40: -50.1917343, -9.1631870, -50.1983681, -9.1557751, -39.1378632, 39.1259308
41: -32.0094223, 7.3297858, -32.0177689, 7.3407021, -38.0456696, 38.0330276
42: -30.1144581, -0.2821140, -30.1195984, -0.2720790, -23.1431503, 23.1463242

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7585241
time: 31.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7702157, upper bound: 13.8064508
time: 34.05 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.8923702, 25.9494095, -7.7911167, 25.8825035, -30.6575203, 30.6444702
1: -0.2537403, 26.8223495, -0.2009339, 26.7403259, -21.2469444, 21.2853394
2: -0.3213663, 25.8950748, -0.2399132, 25.7653122, -20.7572250, 20.8229332
3: -4.6827374, 22.5893230, -4.6057191, 22.4326038, -19.8008118, 19.8687057
4: -7.5307055, 22.5735226, -7.4582276, 22.4647217, -25.4694672, 25.5212555
5: -4.7599936, 24.9091511, -4.6760125, 24.7713318, -23.6514053, 23.7253571
6: -39.4226761, -4.2735939, -39.3748360, -4.3383636, -29.5943146, 29.6125336
7: -9.0846376, 23.4155712, -9.0211515, 23.3833866, -25.6236801, 25.6135979
8: -13.7291765, 20.0359383, -13.6379519, 19.9186573, -29.0752869, 29.1161423
9: -8.4684544, 22.4896488, -8.4432688, 22.4665661, -27.7216034, 27.6254349
10: -28.9334717, 17.8864746, -28.8056831, 17.7991371, -39.5605621, 39.4962311
11: -26.3867855, 6.8433995, -26.1106300, 6.7762494, -29.7765656, 29.5276489
12: -46.1758842, -8.4082670, -46.0034180, -8.5392056, -32.2597809, 32.1843414
13: -32.5184364, 13.2696667, -32.5374336, 13.1721611, -38.3848495, 38.4900055
14: -59.5274200, -2.0902195, -59.2205391, -2.2070770, -56.8482208, 56.6185760
15: -14.1285839, 18.8166924, -14.0551796, 18.7492619, -27.8892975, 27.8523636
16: -15.5968494, 22.2320004, -15.5269356, 22.2446957, -29.5634537, 29.4722214
17: -59.3068733, -6.9125013, -59.0167618, -6.9660664, -51.4227219, 51.1308441
18: -22.1622353, 16.3064079, -21.9408760, 16.2643280, -33.0694504, 32.9559555
19: -22.2412872, 6.0103583, -22.0489388, 5.9599948, -21.9976349, 21.8256836
20: -27.8934593, 0.5756502, -27.7335930, 0.5017223, -22.3280411, 22.2037086
21: -26.4350204, 7.3673658, -26.1743717, 7.2722182, -28.8916855, 28.6901627
22: -29.4980679, 5.2297611, -29.4062023, 5.1938982, -27.4327393, 27.3220520
23: -17.9012337, 12.2620163, -17.7360954, 12.2266445, -24.0458908, 23.8996315
24: -16.5099888, 13.6131859, -16.4239082, 13.5895882, -25.8008652, 25.7363815
25: -23.8156166, 8.8146820, -23.7405033, 8.7820168, -27.8420029, 27.7843018
26: -39.4617691, 4.0126820, -39.2071266, 3.9270401, -33.2629013, 33.0439911
27: -19.5471363, 14.8594742, -19.4276981, 14.8219566, -34.3690948, 34.2871704
28: -21.4474754, 11.4104385, -21.2841454, 11.3589296, -26.2743378, 26.1363640
29: -24.4584293, 6.3592987, -24.3217125, 6.3249259, -28.6175385, 28.4583435
30: -30.1214771, 5.8506608, -29.9254169, 5.7679663, -33.9835052, 33.8568497
31: -23.4238911, 7.6179552, -23.2451553, 7.5771003, -25.0511246, 24.8890228
32: -36.9872818, -2.6859789, -36.9530220, -2.7539873, -29.6669922, 29.6899796
33: -54.4090080, 0.0630369, -54.3494759, -0.1393366, -44.2402039, 44.4237061
34: -49.1090851, -7.0804133, -49.0800552, -7.1469469, -35.7314529, 35.7019958
35: -40.2428741, 3.8138685, -40.2249146, 3.7392387, -35.8406219, 35.8920822
36: -45.5545845, 1.2051497, -45.5360031, 1.1473875, -39.5357819, 39.5280380
37: -60.6799545, -9.9678192, -60.6154404, -10.0008945, -44.7386475, 44.7772369
38: -53.5944519, 3.9025850, -53.5561752, 3.8229313, -48.5407410, 48.6801224
39: -61.8802872, -4.5993395, -61.8547897, -4.7293596, -48.1167755, 48.2347641
40: -50.1694031, -9.1826620, -50.1196671, -9.2248154, -38.9980927, 39.0393066
41: -31.9821320, 7.2164164, -31.9525509, 7.1817336, -37.8504639, 37.8538818
42: -30.0894928, -0.3741961, -30.0838051, -0.4070888, -23.0038338, 23.0026588

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7752795, upper bound: 13.6863122
time: 39.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7842796, upper bound: 13.7338570
time: 30.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.9304485, 25.9719810, -7.7966824, 25.8936691, -30.7064362, 30.6644135
1: -0.2858419, 26.8558960, -0.2034321, 26.7567883, -21.2976418, 21.2980118
2: -0.3464665, 25.9235954, -0.2429802, 25.7795219, -20.7983551, 20.8362160
3: -4.7096553, 22.6258812, -4.6076584, 22.4509964, -19.8478203, 19.8775558
4: -7.5682878, 22.6203461, -7.4622073, 22.4879761, -25.5343666, 25.5506020
5: -4.7906570, 24.9409409, -4.6811357, 24.7873859, -23.7010651, 23.7454376
6: -39.4331627, -4.2624979, -39.3772125, -4.3344107, -29.6113892, 29.6232147
7: -9.1192684, 23.4463730, -9.0268583, 23.3985786, -25.6750183, 25.6280403
8: -13.7581711, 20.0690460, -13.6395969, 19.9341850, -29.1178398, 29.1231804
9: -8.4870710, 22.5153980, -8.4459486, 22.4794807, -27.7558365, 27.6424828
10: -28.9516869, 17.9071617, -28.8081703, 17.8088284, -39.5942535, 39.5066757
11: -26.4190331, 6.8645654, -26.1235657, 6.7777972, -29.7966461, 29.5638351
12: -46.1897812, -8.3848476, -46.0094681, -8.5341005, -32.2745209, 32.2030792
13: -32.5443840, 13.2869129, -32.5492859, 13.1753616, -38.4044952, 38.5038681
14: -59.5528297, -2.0840750, -59.2306938, -2.2054062, -56.8693237, 56.6332703
15: -14.1555061, 18.8484688, -14.0581894, 18.7644768, -27.9336166, 27.8779144
16: -15.6179962, 22.2545433, -15.5307503, 22.2567024, -29.5998535, 29.4760590
17: -59.3569832, -6.8841581, -59.0366859, -6.9637680, -51.4457092, 51.1805420
18: -22.1840248, 16.3124008, -21.9458618, 16.2664776, -33.0927811, 32.9631195
19: -22.2864647, 6.0312071, -22.0705414, 5.9604206, -22.0103073, 21.8674393
20: -27.9329262, 0.5929079, -27.7529259, 0.5024891, -22.3489151, 22.2446251
21: -26.4731636, 7.3868098, -26.1919727, 7.2737103, -28.9029922, 28.7250977
22: -29.5326443, 5.2481399, -29.4222660, 5.1960506, -27.4392624, 27.3532715
23: -17.9439659, 12.2891006, -17.7566795, 12.2292480, -24.0715942, 23.9515343
24: -16.5470409, 13.6292515, -16.4419556, 13.5905495, -25.8259735, 25.7737961
25: -23.8864861, 8.8522196, -23.7756920, 8.7858124, -27.8792496, 27.8617859
26: -39.4926834, 4.0314426, -39.2215805, 3.9303069, -33.2801437, 33.0799484
27: -19.5561943, 14.8646412, -19.4316864, 14.8235283, -34.3797226, 34.2963257
28: -21.4939117, 11.4368191, -21.3069496, 11.3609085, -26.3075638, 26.1896324
29: -24.4833183, 6.3725672, -24.3323536, 6.3264704, -28.6300278, 28.4769516
30: -30.1546745, 5.8741279, -29.9412289, 5.7704353, -34.0126648, 33.8978653
31: -23.4683571, 7.6435986, -23.2659492, 7.5781503, -25.0659714, 24.9386139
32: -36.9970436, -2.6749339, -36.9557838, -2.7501931, -29.6821747, 29.7029953
33: -54.4344368, 0.0826387, -54.3618660, -0.1369257, -44.2449799, 44.4538879
34: -49.1184464, -7.0687103, -49.0817795, -7.1429958, -35.7419968, 35.7168427
35: -40.2590256, 3.8231373, -40.2326469, 3.7407007, -35.8416901, 35.9149170
36: -45.5854492, 1.2260122, -45.5514526, 1.1490698, -39.5427399, 39.5522766
37: -60.7228661, -9.9452190, -60.6359291, -9.9988317, -44.7501373, 44.8098907
38: -53.6230698, 3.9183540, -53.5693207, 3.8259029, -48.5554962, 48.7017136
39: -61.9212990, -4.5720081, -61.8751526, -4.7263269, -48.1281738, 48.2730789
40: -50.1838417, -9.1726370, -50.1227455, -9.2222252, -39.0180817, 39.0486145
41: -31.9962864, 7.2394977, -31.9553623, 7.1923161, -37.8753281, 37.8767090
42: -30.1084480, -0.3555999, -30.0928993, -0.4027510, -23.0233612, 23.0294647

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7752795, upper bound: 13.7072667
time: 35.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7842796, upper bound: 13.7549179
time: 30.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 67.96 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7390582, upper bound: 13.6863122
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7338570
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7072667
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7549179
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7056513
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7532015
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7390582, upper bound: 13.7266048
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7480396, upper bound: 13.7742643
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7610865, upper bound: 13.6868713
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7343751
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7078267
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7554483
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7062105
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7537205
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7610865, upper bound: 13.7271649
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7696890, upper bound: 13.7271649
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7176638
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7654948
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7386184
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7865368
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7369992
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7485743, upper bound: 13.7848401
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7579500
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7485743, upper bound: 13.8058847
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7182276
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7660527
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7391863
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7871037
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7375659
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7702157, upper bound: 13.7853966
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7585241
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7702157, upper bound: 13.8064508
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7752795, upper bound: 13.6863122
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7842796, upper bound: 13.7338570
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7752795, upper bound: 13.7072667
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.96
Output dim: 1, lower bound: -13.7842796, upper bound: 13.7549179
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7855618, upper bound: 13.7544873
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7755462
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7356509
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7567222
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7549967
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8072006, upper bound: 13.7760702
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7667739
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7878132
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7861191
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.7860934, upper bound: 13.8071613
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7673265
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7883757
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7866707
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.96
Output dim: 1, lower bound: -13.8077233, upper bound: 13.8077234

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 55.07 + 3558.03 = 3613.10 seconds
