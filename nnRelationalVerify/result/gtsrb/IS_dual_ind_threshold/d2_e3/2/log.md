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
execution time: IAR + RelationalAnalysis = 2.73 + 50.51 = 53.24 seconds
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 28.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
time: 27.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 55.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 55.46
Output dim: 1, lower bound: -13.7769002, upper bound: 13.8118284
IS_A2, status: Status.UNKNOWN, split count: 1, time: 55.46
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

Time for backsubstitution: 2.09 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 38.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
time: 29.04 seconds

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

Time for backsubstitution: 2.10 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
time: 25.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8106329, upper bound: 13.8106329
time: 28.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 56.05 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 56.05
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 56.05
Output dim: 1, lower bound: -13.7757046, upper bound: 13.8106329
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 56.05
Output dim: 1, lower bound: -13.7751805, upper bound: 13.7789745
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 56.05
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

Time for backsubstitution: 2.10 seconds

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
time: 26.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7749396, upper bound: 13.8098683
time: 28.94 seconds

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

Time for backsubstitution: 2.09 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1759

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7882391, upper bound: 13.8093084
time: 37.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8098681, upper bound: 13.8098683
time: 26.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 66.55 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 66.55
Output dim: 1, lower bound: -13.7533069, upper bound: 13.8093084
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 66.55
Output dim: 1, lower bound: -13.7749396, upper bound: 13.8098683
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 66.55
Output dim: 1, lower bound: -13.7882391, upper bound: 13.8093084
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 66.55
Output dim: 1, lower bound: -13.8098681, upper bound: 13.8098683

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

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
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7507197, upper bound: 13.7886817
time: 36.52 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7507197, upper bound: 13.8080303
time: 38.85 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7723525, upper bound: 13.7892414
time: 36.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
time: 29.64 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

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
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
time: 31.72 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
time: 25.53 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=191, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

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
time: 30.87 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880
time: 31.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 64.95 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7507197, upper bound: 13.7886817
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7507197, upper bound: 13.8080303
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7723525, upper bound: 13.7892414
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7723525, upper bound: 13.8085880
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7869583, upper bound: 13.7886817
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.7869583, upper bound: 13.8080303
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.8085879, upper bound: 13.7892414
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 64.95
Output dim: 1, lower bound: -13.8085879, upper bound: 13.8085880

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

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
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7861191
time: 29.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7498560, upper bound: 13.8071613
time: 32.57 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

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
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7866707
time: 32.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7714884, upper bound: 13.8077234
time: 29.24 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7861191
time: 38.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7860934, upper bound: 13.8071613
time: 30.83 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7673265
time: 31.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7883757
time: 32.13 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=190, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1665

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7866707
time: 86.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8077233, upper bound: 13.8077234
time: 35.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 124.00 seconds
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7493221, upper bound: 13.7861191
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7498560, upper bound: 13.8071613
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7714884, upper bound: 13.7866707
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7714884, upper bound: 13.8077234
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7860934, upper bound: 13.7861191
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.7860934, upper bound: 13.8071613
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7673265
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7883757
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.8077233, upper bound: 13.7866707
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.00
Output dim: 1, lower bound: -13.8077233, upper bound: 13.8077234

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7579500
time: 27.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7485743, upper bound: 13.8058847
time: 37.41 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

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
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7585241
time: 29.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7702157, upper bound: 13.8064508
time: 32.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.0836668, 25.9760056, -8.1219940, 25.9387188, -30.9081841, 30.9691925
1: -0.3787346, 26.8590698, -0.3991699, 26.8058281, -21.4440536, 21.4798393
2: -0.4644160, 25.9277020, -0.4857540, 25.8446941, -20.9826126, 21.0285187
3: -4.8226910, 22.6342354, -4.8382258, 22.5296288, -20.0484505, 20.0378380
4: -7.6957102, 22.6263771, -7.7276192, 22.5507641, -25.7193031, 25.7504005
5: -4.9136004, 24.9480324, -4.9333797, 24.8620014, -23.9037933, 23.9242325
6: -39.4452286, -4.1817627, -39.4353561, -4.1583529, -29.7957077, 29.7667007
7: -9.2195635, 23.4507046, -9.2408953, 23.4331303, -25.8355293, 25.8310699
8: -13.8971615, 20.0778217, -13.9278574, 20.0213757, -29.3455887, 29.3760605
9: -8.5542984, 22.5412750, -8.5949621, 22.5559578, -27.9000473, 27.8401756
10: -29.0284004, 17.9700317, -29.0041122, 17.9737682, -39.8282242, 39.7382507
11: -26.4311657, 6.9491358, -26.2316818, 6.9502730, -29.8798065, 29.7673264
12: -46.1992683, -8.2228432, -46.0991211, -8.1988182, -32.5714417, 32.4574509
13: -32.5674057, 13.3214264, -32.6010094, 13.2623234, -38.5947571, 38.6379242
14: -59.6485710, -1.9793892, -59.5070610, -1.9609852, -57.1314697, 57.0135345
15: -14.2476711, 18.8588943, -14.2576103, 18.8243771, -28.0874252, 28.0524521
16: -15.6731892, 22.2655449, -15.6632347, 22.2988987, -29.6918945, 29.5800171
17: -59.4094200, -6.8197527, -59.2216225, -6.8113270, -51.6369781, 51.4253235
18: -22.2042332, 16.3768921, -22.0583420, 16.3954639, -33.2139587, 33.1276855
19: -22.3052559, 6.1120977, -22.1780243, 6.1249962, -22.1370773, 22.0577240
20: -27.9467945, 0.6943989, -27.8542747, 0.7102084, -22.4910812, 22.4588127
21: -26.4944324, 7.5045414, -26.3293171, 7.5140891, -29.0848618, 28.9912720
22: -29.5462151, 5.3092852, -29.4935207, 5.3243866, -27.5599136, 27.5577202
23: -17.9585342, 12.3565369, -17.8448753, 12.3692379, -24.1680984, 24.0912743
24: -16.5597801, 13.6766853, -16.5019608, 13.6886244, -25.9388428, 25.8730392
25: -23.8959007, 8.9310379, -23.8404808, 8.9510136, -28.0009995, 28.0066147
26: -39.5053635, 4.1705160, -39.3442459, 4.2065258, -33.4759903, 33.3468246
27: -19.5825081, 14.9189301, -19.5177746, 14.9349098, -34.5174179, 34.4367065
28: -21.5114613, 11.5285368, -21.4026661, 11.5493279, -26.4558029, 26.3733292
29: -24.4972591, 6.4249945, -24.4108181, 6.4381390, -28.7374878, 28.6593170
30: -30.1643963, 5.9617877, -30.0227451, 5.9573259, -34.1693192, 34.0616989
31: -23.4895096, 7.7219496, -23.3739090, 7.7400188, -25.2008286, 25.1088753
32: -37.0083084, -2.5959420, -37.0059433, -2.5826855, -29.8406296, 29.8416443
33: -54.4870758, 0.1706734, -54.5167770, 0.0859365, -44.4664459, 44.6371536
34: -49.1456146, -7.0074253, -49.1728859, -6.9931812, -35.8939972, 35.8496857
35: -40.2822418, 3.8944798, -40.3223000, 3.9015369, -35.9964371, 36.0562210
36: -45.5969429, 1.3170013, -45.6269569, 1.3434916, -39.7160797, 39.7515259
37: -60.7408180, -9.8717995, -60.7339745, -9.8453617, -44.8845978, 44.9745560
38: -53.6408195, 4.0330372, -53.6777153, 4.0707273, -48.7643738, 48.9036865
39: -61.9404526, -4.4908848, -61.9661713, -4.5431137, -48.2884216, 48.4532089
40: -50.2070160, -9.1518946, -50.1940956, -9.1680450, -39.1213684, 39.1513290
41: -32.0135765, 7.2951870, -32.0187187, 7.3119769, -38.0149765, 37.9993057
42: -30.1188927, -0.3087044, -30.1207294, -0.2988181, -23.1328087, 23.1203804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

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
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7757449, upper bound: 13.7579500
time: 32.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7848125, upper bound: 13.8058847
time: 21.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1151476, 25.9758396, -8.0582104, 25.9094658, -30.9073067, 30.9364700
1: -0.3993392, 26.8433552, -0.3703089, 26.7650814, -21.4040337, 21.4617081
2: -0.4486084, 25.9186249, -0.3965240, 25.7910309, -20.8937073, 20.9804459
3: -4.7842579, 22.6098385, -4.7299204, 22.4628010, -19.9199219, 20.0015373
4: -7.6990190, 22.5915833, -7.6585479, 22.4861393, -25.6363869, 25.7153511
5: -4.8790636, 24.9350243, -4.8150482, 24.7980766, -23.7803497, 23.8729095
6: -39.4412613, -4.1306438, -39.4171677, -4.1502256, -29.8178864, 29.7994919
7: -9.2404060, 23.4477348, -9.1858063, 23.4038525, -25.7731476, 25.7915688
8: -13.9131031, 20.0632172, -13.8773251, 19.9643402, -29.2869949, 29.3465118
9: -8.6201448, 22.5253448, -8.6110420, 22.4967346, -27.8849258, 27.8150444
10: -29.0857849, 17.9282036, -28.9931755, 17.8544407, -39.7505951, 39.6991425
11: -26.4343395, 6.9013376, -26.1581802, 6.8663964, -29.9190292, 29.6349640
12: -46.2012177, -8.2777872, -46.0211105, -8.3961926, -32.4376373, 32.3443604
13: -32.5837593, 13.3115492, -32.5956306, 13.2263145, -38.5986710, 38.6274872
14: -59.6908035, -2.0515432, -59.4252243, -2.1532040, -57.0745697, 56.8902283
15: -14.2393894, 18.8427925, -14.2217712, 18.7969074, -28.0484695, 28.0470238
16: -15.7365265, 22.2744236, -15.6597500, 22.2662582, -29.6903915, 29.6070633
17: -59.4180717, -6.8789959, -59.1441574, -6.9183578, -51.6108093, 51.3519897
18: -22.2069111, 16.3740158, -21.9959431, 16.3459473, -33.2109985, 33.1127853
19: -22.2729568, 6.0909448, -22.1011410, 6.0731888, -22.1342316, 21.9499435
20: -27.9230537, 0.6750603, -27.7823448, 0.6366458, -22.4760475, 22.3437233
21: -26.4728050, 7.4634032, -26.2427349, 7.4182539, -29.0636215, 28.8508034
22: -29.5435181, 5.3288617, -29.4587631, 5.3099337, -27.5697556, 27.4506073
23: -17.9282761, 12.3377657, -17.7727966, 12.3194561, -24.1530075, 23.9984818
24: -16.5363197, 13.6860514, -16.4694023, 13.6862679, -25.9144211, 25.8455200
25: -23.8474407, 8.9220209, -23.7807655, 8.9238377, -27.9932251, 27.9169769
26: -39.5111542, 4.1331034, -39.2483521, 4.0534768, -33.4332352, 33.1871643
27: -19.5899601, 14.9452419, -19.4890461, 14.9364843, -34.5264435, 34.4342880
28: -21.4802437, 11.5200825, -21.3275623, 11.5003309, -26.4398499, 26.2745972
29: -24.4967461, 6.4345994, -24.3691349, 6.4138966, -28.7233429, 28.5677109
30: -30.1397266, 5.9300294, -29.9711437, 5.8988376, -34.1334229, 33.9867172
31: -23.4591370, 7.7143135, -23.2991543, 7.7094655, -25.2150421, 25.0371742
32: -37.0088348, -2.5845881, -36.9851227, -2.6252446, -29.8181381, 29.8293152
33: -54.4603691, 0.2147779, -54.4226494, 0.0553169, -44.4373474, 44.6142960
34: -49.1624413, -6.9343662, -49.1335754, -6.9885864, -35.9310837, 35.8820343
35: -40.2979774, 3.9681606, -40.2890282, 3.9237556, -36.0464325, 36.0818405
36: -45.6134758, 1.3808899, -45.6008453, 1.3563347, -39.7598648, 39.7532806
37: -60.7362518, -9.8473339, -60.6847610, -9.8628674, -44.8917084, 44.9296188
38: -53.6765366, 4.1310787, -53.6410294, 4.0886316, -48.8235321, 48.9673615
39: -61.9331856, -4.4722109, -61.9347763, -4.5629444, -48.2880554, 48.4350739
40: -50.2062378, -9.1350727, -50.1667480, -9.1760521, -39.0885315, 39.1562271
41: -32.0196915, 7.3261032, -31.9938221, 7.3103299, -38.0234528, 38.0075684
42: -30.1119480, -0.2815547, -30.0991325, -0.3066368, -23.1289215, 23.1174011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7182276
time: 30.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7660527
time: 33.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1532326, 25.9984283, -8.0637960, 25.9206276, -30.9562302, 30.9564056
1: -0.4314270, 26.8769093, -0.3728766, 26.7815437, -21.4547157, 21.4743805
2: -0.4736919, 25.9471474, -0.3996096, 25.8052540, -20.9348526, 20.9936981
3: -4.8112073, 22.6464329, -4.7319026, 22.4812431, -19.9669380, 20.0103951
4: -7.7365646, 22.6384430, -7.6624775, 22.5093613, -25.7012634, 25.7446747
5: -4.9096632, 24.9668064, -4.8201804, 24.8141365, -23.8299828, 23.8929749
6: -39.4517441, -4.1195421, -39.4195404, -4.1462936, -29.8349609, 29.8101044
7: -9.2750816, 23.4785690, -9.1914492, 23.4190388, -25.8244820, 25.8060074
8: -13.9420509, 20.0963516, -13.8789692, 19.9798965, -29.3295364, 29.3535461
9: -8.6387072, 22.5511322, -8.6137075, 22.5096169, -27.9191208, 27.8321304
10: -29.1039696, 17.9489632, -28.9957161, 17.8641396, -39.7843475, 39.7095795
11: -26.4665337, 6.9224544, -26.1711006, 6.8679314, -29.9390564, 29.6711311
12: -46.2151222, -8.2544384, -46.0271797, -8.3910761, -32.4523315, 32.3630524
13: -32.6096649, 13.3287754, -32.6074753, 13.2294722, -38.6183701, 38.6413498
14: -59.7162209, -2.0453682, -59.4353104, -2.1516247, -57.0956116, 56.9048157
15: -14.2663021, 18.8745995, -14.2248077, 18.8121414, -28.0928345, 28.0725937
16: -15.7576389, 22.2969685, -15.6635895, 22.2782345, -29.7267685, 29.6108704
17: -59.4680290, -6.8507414, -59.1639977, -6.9160271, -51.6337357, 51.4016113
18: -22.2286682, 16.3799858, -22.0009441, 16.3480721, -33.2343521, 33.1199265
19: -22.3181839, 6.1117992, -22.1227551, 6.0736194, -22.1469154, 21.9917259
20: -27.9625931, 0.6922889, -27.8016739, 0.6374273, -22.4969177, 22.3846169
21: -26.5109596, 7.4828444, -26.2603111, 7.4197516, -29.0749817, 28.8857117
22: -29.5781269, 5.3472905, -29.4748402, 5.3121452, -27.5762939, 27.4818268
23: -17.9709892, 12.3648987, -17.7933884, 12.3220387, -24.1787109, 24.0504150
24: -16.5733528, 13.7021236, -16.4874058, 13.6872482, -25.9395294, 25.8829651
25: -23.9183464, 8.9595127, -23.8159256, 8.9276285, -28.0304184, 27.9944496
26: -39.5420456, 4.1518855, -39.2628403, 4.0567875, -33.4504929, 33.2231445
27: -19.5989990, 14.9504585, -19.4930420, 14.9380608, -34.5370598, 34.4435005
28: -21.5266800, 11.5465145, -21.3503914, 11.5022860, -26.4730530, 26.3278503
29: -24.5216217, 6.4479122, -24.3798180, 6.4154654, -28.7358322, 28.5862503
30: -30.1729565, 5.9534645, -29.9869423, 5.9012733, -34.1626282, 34.0277405
31: -23.5036507, 7.7399311, -23.3199978, 7.7105341, -25.2299042, 25.0867996
32: -37.0185928, -2.5735688, -36.9878845, -2.6214724, -29.8332596, 29.8423309
33: -54.4858360, 0.2344818, -54.4350700, 0.0577183, -44.4421539, 44.6445312
34: -49.1718483, -6.9226279, -49.1353302, -6.9846711, -35.9416046, 35.8968811
35: -40.3141785, 3.9774895, -40.2968178, 3.9251871, -36.0475159, 36.1046448
36: -45.6444092, 1.4017153, -45.6163101, 1.3579369, -39.7667542, 39.7775879
37: -60.7791977, -9.8247147, -60.7052765, -9.8608551, -44.9031219, 44.9623108
38: -53.7051849, 4.1468143, -53.6542130, 4.0916748, -48.8382416, 48.9889603
39: -61.9741974, -4.4449272, -61.9551086, -4.5598402, -48.2993774, 48.4734344
40: -50.2206459, -9.1251202, -50.1698608, -9.1735306, -39.1085815, 39.1654663
41: -32.0338020, 7.3491826, -31.9966450, 7.3208880, -38.0482788, 38.0303116
42: -30.1308689, -0.2629642, -30.1082382, -0.3023233, -23.1484413, 23.1442299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7391863
time: 25.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7871037
time: 30.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1718445, 25.9767876, -8.1794643, 25.9294739, -30.9864273, 31.0365448
1: -0.4304180, 26.8444901, -0.4385276, 26.7901134, -21.4647522, 21.5288391
2: -0.5077386, 25.9205971, -0.5174308, 25.8315353, -20.9937057, 21.0711555
3: -4.8505344, 22.6118603, -4.8643761, 22.5133553, -20.0443192, 20.0750885
4: -7.7530136, 22.5945034, -7.7711287, 22.5288849, -25.7303925, 25.7857819
5: -4.9519830, 24.9387798, -4.9638157, 24.8484802, -23.9074554, 23.9641266
6: -39.4477158, -4.1219435, -39.4371033, -4.1273632, -29.8453903, 29.8325882
7: -9.2858944, 23.4501801, -9.2867374, 23.4195290, -25.8569374, 25.9000092
8: -13.9587908, 20.0672379, -13.9727793, 20.0087547, -29.3793373, 29.4391441
9: -8.6293478, 22.5466690, -8.6382513, 22.5444946, -27.9406662, 27.9012070
10: -29.0917835, 17.9816742, -29.0431023, 17.9688435, -39.8556442, 39.7999039
11: -26.4380417, 6.9471669, -26.2380886, 6.9575448, -29.9166870, 29.7700958
12: -46.2042847, -8.1629219, -46.0965309, -8.1628437, -32.6124802, 32.5333862
13: -32.5930367, 13.3270187, -32.6148491, 13.2670631, -38.6380615, 38.7029572
14: -59.7098694, -1.9550819, -59.5394211, -1.9588203, -57.2059174, 57.0997162
15: -14.2619190, 18.8451824, -14.2730732, 18.8192348, -28.0958252, 28.0702896
16: -15.7506027, 22.2837429, -15.7075253, 22.2880592, -29.7154160, 29.6664352
17: -59.4253616, -6.8245516, -59.2343178, -6.8052082, -51.6906891, 51.4933777
18: -22.2121716, 16.4079895, -22.0619240, 16.4117680, -33.2402191, 33.1626816
19: -22.2822762, 6.1253042, -22.1626816, 6.1419954, -22.1663513, 22.0516472
20: -27.9306431, 0.7221479, -27.8384991, 0.7319961, -22.5164948, 22.4595985
21: -26.4814930, 7.5186434, -26.3204136, 7.5295262, -29.1175995, 28.9963760
22: -29.5502377, 5.3469415, -29.4842854, 5.3512511, -27.6217728, 27.5605659
23: -17.9360867, 12.3704920, -17.8286667, 12.3868275, -24.1845856, 24.0723648
24: -16.5426769, 13.6946430, -16.4882011, 13.7048054, -25.9509430, 25.8654633
25: -23.8515816, 8.9461088, -23.8087559, 8.9741058, -28.0184250, 27.9602051
26: -39.5163383, 4.2285271, -39.3350449, 4.2421255, -33.5396271, 33.3691406
27: -19.6029034, 14.9540443, -19.5237408, 14.9534054, -34.5563087, 34.4777832
28: -21.4912567, 11.5570946, -21.3841362, 11.5757694, -26.4799118, 26.3648643
29: -24.5037079, 6.4553804, -24.4087429, 6.4591613, -28.7798996, 28.6752434
30: -30.1433525, 5.9602957, -30.0118847, 5.9648499, -34.1685333, 34.0515747
31: -23.4704609, 7.7393923, -23.3601398, 7.7605176, -25.2363052, 25.1103745
32: -37.0135422, -2.5540442, -37.0074310, -2.5601072, -29.8789139, 29.8880081
33: -54.5028076, 0.2262802, -54.5102234, 0.1212978, -44.5447998, 44.6494904
34: -49.1807556, -6.9285464, -49.1744957, -6.9520884, -35.9735107, 35.9086456
35: -40.3118248, 3.9700546, -40.3189163, 3.9429770, -36.0829010, 36.0907669
36: -45.6183052, 1.3938227, -45.6162376, 1.3909826, -39.8110962, 39.8139572
37: -60.7430344, -9.8234406, -60.7194977, -9.8109493, -44.9571838, 44.9801407
38: -53.6829834, 4.1481800, -53.6712151, 4.1324968, -48.8878479, 48.9973297
39: -61.9416809, -4.4576445, -61.9561920, -4.5159473, -48.3506927, 48.4747467
40: -50.2161331, -9.1297636, -50.1995010, -9.1550312, -39.1463318, 39.1857071
41: -32.0287743, 7.3356318, -32.0207138, 7.3331008, -38.0587616, 38.0448837
42: -30.1171074, -0.2676105, -30.1142006, -0.2733164, -23.1561813, 23.1536636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

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

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7375659
time: 32.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7853966
time: 24.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.2099276, 25.9993553, -8.1850719, 25.9406300, -31.0353165, 31.0564919
1: -0.4625292, 26.8780174, -0.4410458, 26.8065796, -21.5154495, 21.5415535
2: -0.5328269, 25.9491062, -0.5205665, 25.8457680, -21.0348358, 21.0844269
3: -4.8774686, 22.6484451, -4.8663421, 22.5317593, -20.0913353, 20.0839729
4: -7.7905998, 22.6413479, -7.7750740, 22.5521011, -25.7952805, 25.8151360
5: -4.9826193, 24.9705868, -4.9688892, 24.8645439, -23.9570961, 23.9841805
6: -39.4581642, -4.1109180, -39.4394760, -4.1234818, -29.8624039, 29.8432159
7: -9.3204679, 23.4809914, -9.2924252, 23.4347134, -25.9082527, 25.9144363
8: -13.9877768, 20.1004143, -13.9744673, 20.0242958, -29.4218903, 29.4462166
9: -8.6479197, 22.5724258, -8.6409130, 22.5574036, -27.9748840, 27.9182587
10: -29.1099777, 18.0024395, -29.0456352, 17.9785118, -39.8894348, 39.8103485
11: -26.4702530, 6.9683113, -26.2510262, 6.9590931, -29.9367371, 29.8062897
12: -46.2182083, -8.1396027, -46.1025696, -8.1577168, -32.6272583, 32.5520554
13: -32.6188812, 13.3442039, -32.6266670, 13.2703152, -38.6577301, 38.7168350
14: -59.7351685, -1.9489326, -59.5495453, -1.9572248, -57.2269897, 57.1143646
15: -14.2888393, 18.8770180, -14.2761250, 18.8344536, -28.1401749, 28.0958862
16: -15.7717190, 22.3062248, -15.7113228, 22.3000183, -29.7517853, 29.6702499
17: -59.4752846, -6.7962627, -59.2542267, -6.8028555, -51.7136993, 51.5429993
18: -22.2340050, 16.4140186, -22.0668907, 16.4139214, -33.2635727, 33.1698380
19: -22.3275223, 6.1461945, -22.1842957, 6.1424379, -22.1790619, 22.0934372
20: -27.9701328, 0.7393851, -27.8578568, 0.7327833, -22.5373611, 22.5004807
21: -26.5196285, 7.5380745, -26.3380356, 7.5310311, -29.1289520, 29.0312576
22: -29.5848122, 5.3653712, -29.5003586, 5.3533988, -27.6283722, 27.5917816
23: -17.9787827, 12.3976040, -17.8492584, 12.3894577, -24.2103119, 24.1242599
24: -16.5797005, 13.7107315, -16.5061836, 13.7057590, -25.9760666, 25.9029160
25: -23.9224796, 8.9836016, -23.8439198, 8.9779081, -28.0556564, 28.0376854
26: -39.5472374, 4.2472601, -39.3495064, 4.2454152, -33.5568848, 33.4050598
27: -19.6119442, 14.9591837, -19.5277081, 14.9549980, -34.5669403, 34.4868927
28: -21.5376740, 11.5835438, -21.4069500, 11.5777264, -26.5131149, 26.4181099
29: -24.5286255, 6.4686847, -24.4194145, 6.4607201, -28.7923508, 28.6938095
30: -30.1765938, 5.9837632, -30.0277290, 5.9673185, -34.1977463, 34.0926285
31: -23.5149574, 7.7650433, -23.3809566, 7.7615776, -25.2512283, 25.1599884
32: -37.0232964, -2.5430527, -37.0101852, -2.5563221, -29.8940964, 29.9009933
33: -54.5283356, 0.2460003, -54.5226135, 0.1236677, -44.5495911, 44.6797485
34: -49.1901932, -6.9167681, -49.1762009, -6.9481287, -35.9839935, 35.9234924
35: -40.3280525, 3.9793606, -40.3266640, 3.9444447, -36.0839081, 36.1136246
36: -45.6492844, 1.4147072, -45.6317139, 1.3926048, -39.8181000, 39.8382568
37: -60.7860184, -9.8007946, -60.7399788, -9.8089371, -44.9686584, 45.0129013
38: -53.7116241, 4.1639185, -53.6843491, 4.1355152, -48.9024963, 49.0189056
39: -61.9827232, -4.4302759, -61.9764862, -4.5128708, -48.3620605, 48.5129700
40: -50.2305222, -9.1197977, -50.2025948, -9.1524544, -39.1663361, 39.1950073
41: -32.0428925, 7.3586946, -32.0235214, 7.3436384, -38.0836792, 38.0676956
42: -30.1360168, -0.2490625, -30.1233082, -0.2689905, -23.1756973, 23.1804466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=190, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7585241
time: 32.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.8064510, upper bound: 13.8064508
time: 35.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 70.03 seconds
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7395232, upper bound: 13.7579500
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7485743, upper bound: 13.8058847
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7615537, upper bound: 13.7585241
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7702157, upper bound: 13.8064508
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7757449, upper bound: 13.7579500
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7848125, upper bound: 13.8058847
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7182276
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7660527
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7391863
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7871037
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7375659
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.8064510, upper bound: 13.7853966
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.7977739, upper bound: 13.7585241
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 70.03
Output dim: 1, lower bound: -13.8064510, upper bound: 13.8064508

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.9663057, 25.9146843, -8.0784140, 25.9377270, -30.8055496, 30.8596573
1: -0.3041167, 26.7841434, -0.3731503, 26.8046684, -21.3765068, 21.3553314
2: -0.3564448, 25.8190384, -0.4416504, 25.8425026, -20.8858376, 20.8468475
3: -4.7013173, 22.5159607, -4.7878528, 22.5288830, -19.9674301, 19.8790512
4: -7.5990968, 22.5298691, -7.6885662, 22.5480995, -25.6340408, 25.5770721
5: -4.7876549, 24.8381424, -4.8819427, 24.8603916, -23.8059616, 23.7387314
6: -39.4157639, -4.2025118, -39.4306183, -4.1598501, -29.7548218, 29.7585754
7: -9.1277704, 23.4005165, -9.2090759, 23.4317665, -25.7518730, 25.7001877
8: -13.8318834, 19.9922543, -13.9014864, 20.0175686, -29.2787628, 29.2299538
9: -8.5298271, 22.5194340, -8.5859547, 22.5528679, -27.8280716, 27.7985611
10: -28.9557915, 17.8830185, -28.9979019, 17.9481773, -39.7071915, 39.6103897
11: -26.2026482, 6.8363619, -26.2270336, 6.9023013, -29.6028900, 29.6927567
12: -46.0785255, -8.3800020, -46.0969391, -8.2620153, -32.3962479, 32.3291245
13: -32.5344048, 13.2309313, -32.5925789, 13.2540169, -38.5729141, 38.5503311
14: -59.4333687, -2.1359749, -59.4925003, -2.0267220, -56.8942719, 56.8933105
15: -14.2146397, 18.8108673, -14.2565861, 18.8221283, -28.0370331, 28.0174408
16: -15.5880489, 22.2523212, -15.6500874, 22.3053303, -29.6031494, 29.4966583
17: -59.1730499, -6.9307079, -59.2130699, -6.8580093, -51.4230423, 51.3404083
18: -22.0281239, 16.2950058, -22.0543900, 16.3598938, -33.0644913, 33.0797501
19: -22.1476974, 6.0238581, -22.1715927, 6.0866013, -21.9184380, 21.9874268
20: -27.8215847, 0.6008120, -27.8489170, 0.6703939, -22.3081589, 22.3841515
21: -26.2964573, 7.3846045, -26.3221436, 7.4628539, -28.8180695, 28.8893509
22: -29.4476585, 5.2571173, -29.4877815, 5.3042498, -27.3848114, 27.5026817
23: -17.8163986, 12.2677174, -17.8393593, 12.3326826, -23.9580688, 24.0197754
24: -16.4770203, 13.6162653, -16.4983711, 13.6623878, -25.8336639, 25.8266983
25: -23.8085079, 8.8643742, -23.8373642, 8.9227057, -27.8390427, 27.9553833
26: -39.2979507, 4.0041370, -39.3401527, 4.1330433, -33.1546860, 33.2157440
27: -19.4785213, 14.8510227, -19.5091019, 14.9044209, -34.3829422, 34.3601227
28: -21.3626518, 11.4278965, -21.3947811, 11.5060024, -26.2288513, 26.2906990
29: -24.3714218, 6.3717194, -24.4045906, 6.4166608, -28.5287704, 28.6019821
30: -30.0049992, 5.8502679, -30.0187416, 5.9140821, -33.9664307, 33.9664536
31: -23.3391418, 7.6460094, -23.3664589, 7.7066989, -25.0043869, 25.0512543
32: -36.9849243, -2.6306152, -37.0017014, -2.5911331, -29.8028412, 29.8209763
33: -54.4040146, 0.0300655, -54.4831543, 0.0751209, -44.3429565, 44.4492569
34: -49.1199417, -7.0476594, -49.1679764, -6.9993992, -35.8022766, 35.8203506
35: -40.2661743, 3.8574810, -40.3200607, 3.8977261, -35.9146729, 36.0090103
36: -45.5709763, 1.2848034, -45.6235199, 1.3357258, -39.6274414, 39.7139664
37: -60.6875305, -9.9099026, -60.7308273, -9.8552656, -44.7929382, 44.9071198
38: -53.6030579, 3.9944916, -53.6738396, 4.0638542, -48.7086029, 48.8095932
39: -61.9033432, -4.5951538, -61.9527969, -4.5542622, -48.2277527, 48.3292999
40: -50.1682129, -9.1952391, -50.1886063, -9.1723537, -39.0911560, 39.0799484
41: -31.9801006, 7.2662630, -32.0125732, 7.3078828, -37.9757690, 37.9722366
42: -30.0972977, -0.3417673, -30.1168690, -0.3028250, -23.0895462, 23.0859985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6997229, upper bound: 13.7970446
time: 39.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.6997229, upper bound: 13.8058849
time: 22.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.0925388, 25.9380302, -8.1414871, 25.9396133, -30.9326668, 30.9469299
1: -0.3878756, 26.8030930, -0.4150457, 26.8054352, -21.4478874, 21.4170303
2: -0.4248409, 25.8403969, -0.4764404, 25.8435631, -20.9380264, 20.9027557
3: -4.7561107, 22.5301285, -4.8159571, 22.5310173, -20.0102768, 19.9262314
4: -7.6939259, 22.5448494, -7.7359791, 22.5494633, -25.7100258, 25.6426697
5: -4.8565655, 24.8606339, -4.9174342, 24.8629303, -23.8592491, 23.7990875
6: -39.4286766, -4.1316462, -39.4346924, -4.1249771, -29.8215256, 29.8351440
7: -9.2286654, 23.4308224, -9.2605419, 23.4333496, -25.8246078, 25.7843170
8: -13.9224977, 20.0148430, -13.9480515, 20.0205059, -29.3550682, 29.3000679
9: -8.6234856, 22.5505486, -8.6318941, 22.5543098, -27.9029465, 27.8766251
10: -29.0373993, 17.9153938, -29.0394535, 17.9528542, -39.7683716, 39.6828995
11: -26.2417431, 6.8555422, -26.2463703, 6.9110966, -29.6597748, 29.7313576
12: -46.0974846, -8.2967758, -46.1003799, -8.2209482, -32.4521713, 32.4237328
13: -32.5858498, 13.2536554, -32.6181755, 13.2620716, -38.6358795, 38.6292648
14: -59.5200844, -2.1055584, -59.5348091, -2.0228863, -56.9900055, 56.9942169
15: -14.2558479, 18.8290195, -14.2750568, 18.8321934, -28.0898743, 28.0608368
16: -15.6866264, 22.2929974, -15.6981735, 22.3064384, -29.6630173, 29.5877151
17: -59.2388191, -6.9072294, -59.2461281, -6.8495884, -51.4996414, 51.4581146
18: -22.0578384, 16.3321476, -22.0628738, 16.3783073, -33.1140900, 33.1219254
19: -22.1699619, 6.0579143, -22.1779251, 6.1040211, -21.9603958, 22.0231209
20: -27.8449516, 0.6457896, -27.8525085, 0.6929455, -22.3544540, 22.4258385
21: -26.3216896, 7.4181409, -26.3308697, 7.4797959, -28.8623428, 28.9293022
22: -29.4861908, 5.3131795, -29.4946480, 5.3333325, -27.4536667, 27.5367050
23: -17.8366089, 12.3087788, -17.8437767, 12.3528805, -24.0003662, 24.0527687
24: -16.4969711, 13.6503038, -16.5026226, 13.6795311, -25.8707733, 25.8565216
25: -23.8351440, 8.9170017, -23.8407555, 8.9496031, -27.8937149, 27.9864197
26: -39.3398285, 4.0808916, -39.3453865, 4.1719136, -33.2357483, 33.2740173
27: -19.5079422, 14.8912849, -19.5190239, 14.9244490, -34.4323921, 34.4103088
28: -21.3888512, 11.4829216, -21.3990669, 11.5343809, -26.2861786, 26.3355179
29: -24.4027119, 6.4153976, -24.4131546, 6.4392424, -28.5844040, 28.6364594
30: -30.0171986, 5.8722420, -30.0237083, 5.9240408, -33.9950104, 33.9973755
31: -23.3646393, 7.6890574, -23.3735447, 7.7282438, -25.0547485, 25.1023712
32: -36.9999428, -2.5776944, -37.0059586, -2.5648408, -29.8562775, 29.8802795
33: -54.4452324, 0.1054325, -54.4890289, 0.1129551, -44.4263306, 44.4919205
34: -49.1643906, -6.9570580, -49.1713104, -6.9542809, -35.8922882, 35.8940811
35: -40.3119850, 3.9422836, -40.3245010, 3.9406796, -36.0028305, 36.0663528
36: -45.6232948, 1.3824186, -45.6282349, 1.3848591, -39.7294312, 39.8007507
37: -60.7327423, -9.8389664, -60.7368507, -9.8188305, -44.8758392, 44.9454269
38: -53.6739120, 4.1254158, -53.6805649, 4.1287098, -48.8466797, 48.9248047
39: -61.9455185, -4.5344849, -61.9631348, -4.5240412, -48.3017273, 48.3891525
40: -50.1917343, -9.1631870, -50.1971245, -9.1567860, -39.1360626, 39.1236115
41: -32.0094223, 7.3297858, -32.0173721, 7.3395920, -38.0444031, 38.0406265
42: -30.1144581, -0.2821140, -30.1194038, -0.2730246, -23.1320343, 23.1460876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7223054, upper bound: 13.7977740
time: 29.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7223054, upper bound: 13.8064511
time: 27.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.0836668, 25.9760056, -8.1184635, 25.9385872, -30.9089966, 30.9655685
1: -0.3787346, 26.8590698, -0.3968201, 26.8057747, -21.4440041, 21.4549561
2: -0.4644160, 25.9277020, -0.4836369, 25.8446045, -20.9825363, 20.9972572
3: -4.8226910, 22.6342354, -4.8363881, 22.5295162, -20.0482674, 20.0004311
4: -7.6957102, 22.6263771, -7.7246590, 22.5506210, -25.7191086, 25.7106628
5: -4.9136004, 24.9480324, -4.9312105, 24.8617859, -23.9035416, 23.8881149
6: -39.4452286, -4.1817627, -39.4350815, -4.1593876, -29.7935867, 29.7839508
7: -9.2195635, 23.4507046, -9.2379856, 23.4329796, -25.8353577, 25.7759171
8: -13.8971615, 20.0778217, -13.9249048, 20.0211849, -29.3453903, 29.3413391
9: -8.5542984, 22.5412750, -8.5923414, 22.5558090, -27.8999252, 27.8082504
10: -29.0284004, 17.9700317, -29.0016594, 17.9734421, -39.8278961, 39.7019196
11: -26.4311657, 6.9491358, -26.2302780, 6.9493876, -29.8785858, 29.7737350
12: -46.1992683, -8.2228432, -46.0988464, -8.2004566, -32.5740585, 32.4562836
13: -32.5674057, 13.3214264, -32.5994644, 13.2617168, -38.6154022, 38.6360168
14: -59.6485710, -1.9793892, -59.5051498, -1.9612684, -57.1605835, 57.0112152
15: -14.2476711, 18.8588943, -14.2563190, 18.8238220, -28.0881271, 28.0508347
16: -15.6731892, 22.2655449, -15.6606369, 22.2987919, -29.6917725, 29.5200233
17: -59.4094200, -6.8197527, -59.2192230, -6.8128357, -51.7142639, 51.4223633
18: -22.2042332, 16.3768921, -22.0578098, 16.3943386, -33.2030563, 33.1270905
19: -22.3052559, 6.1120977, -22.1776543, 6.1235399, -22.1165237, 22.0574150
20: -27.9467945, 0.6943989, -27.8540173, 0.7087698, -22.4805756, 22.4585686
21: -26.4944324, 7.5045414, -26.3288651, 7.5126662, -29.0706406, 28.9907455
22: -29.5462151, 5.3092852, -29.4930553, 5.3224916, -27.5082779, 27.5571938
23: -17.9585342, 12.3565369, -17.8445511, 12.3676987, -24.1380234, 24.0909576
24: -16.5597801, 13.6766853, -16.5015907, 13.6872406, -25.9207001, 25.8726883
25: -23.8959007, 8.9310379, -23.8402863, 8.9489412, -27.9493561, 28.0063133
26: -39.5053635, 4.1705160, -39.3438568, 4.2043920, -33.4410172, 33.3464584
27: -19.5825081, 14.9189301, -19.5171127, 14.9335947, -34.5161018, 34.4360428
28: -21.5114613, 11.5285368, -21.4023209, 11.5472622, -26.4206619, 26.3729820
29: -24.4972591, 6.4249945, -24.4103298, 6.4365301, -28.6965179, 28.6587448
30: -30.1643963, 5.9617877, -30.0223389, 5.9561014, -34.1669312, 34.0605240
31: -23.4895096, 7.7219496, -23.3734913, 7.7383451, -25.1902313, 25.1085281
32: -37.0083084, -2.5959420, -37.0055733, -2.5839987, -29.8391953, 29.8596344
33: -54.4870758, 0.1706734, -54.5163345, 0.0837097, -44.4071503, 44.6366043
34: -49.1456146, -7.0074253, -49.1725807, -6.9955111, -35.8640900, 35.8493576
35: -40.2822418, 3.8944798, -40.3219261, 3.8990879, -35.9451294, 36.0558548
36: -45.5969429, 1.3170013, -45.6265831, 1.3408337, -39.6893845, 39.7511749
37: -60.7408180, -9.8717995, -60.7334900, -9.8465080, -44.8297424, 44.9739456
38: -53.6408195, 4.0330372, -53.6772652, 4.0672264, -48.7239990, 48.9032288
39: -61.9404526, -4.4908848, -61.9654694, -4.5441990, -48.2693481, 48.4524155
40: -50.2070160, -9.1518946, -50.1928406, -9.1689720, -39.1195831, 39.1490326
41: -32.0135765, 7.2951870, -32.0183372, 7.3108134, -38.0136719, 38.0068741
42: -30.1188927, -0.3087044, -30.1205502, -0.2997317, -23.1221161, 23.1201477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7359460, upper bound: 13.7970446
time: 27.21 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7359460, upper bound: 13.8058849
time: 26.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1151476, 25.9758396, -8.0546322, 25.9093323, -30.9081116, 30.9328537
1: -0.3993392, 26.8433552, -0.3679714, 26.7650146, -21.4039612, 21.4368248
2: -0.4486084, 25.9186249, -0.3943806, 25.7909470, -20.8936119, 20.9491272
3: -4.7842579, 22.6098385, -4.7280269, 22.4626694, -19.9197121, 19.9651833
4: -7.6990190, 22.5915833, -7.6555853, 22.4860115, -25.6362076, 25.6764908
5: -4.8790636, 24.9350243, -4.8128943, 24.7978706, -23.7800789, 23.8372192
6: -39.4412613, -4.1306438, -39.4169006, -4.1513109, -29.8157349, 29.8167267
7: -9.2404060, 23.4477348, -9.1828403, 23.4036942, -25.7729721, 25.7371635
8: -13.9131031, 20.0632172, -13.8743811, 19.9641495, -29.2867661, 29.3117943
9: -8.6201448, 22.5253448, -8.6084738, 22.4966412, -27.8847809, 27.7831192
10: -29.0857849, 17.9282036, -28.9906273, 17.8541451, -39.7502747, 39.6632233
11: -26.4343395, 6.9013376, -26.1568146, 6.8654985, -29.9178085, 29.6410294
12: -46.2012177, -8.2777872, -46.0208664, -8.3979445, -32.4403915, 32.3432007
13: -32.5837593, 13.3115492, -32.5940132, 13.2257538, -38.6192627, 38.6255264
14: -59.6908035, -2.0515432, -59.4230499, -2.1534882, -57.1039276, 56.8878174
15: -14.2393894, 18.8427925, -14.2204351, 18.7963791, -28.0491791, 28.0453720
16: -15.7365265, 22.2744236, -15.6572018, 22.2661381, -29.6902542, 29.5478973
17: -59.4180717, -6.8789959, -59.1422195, -6.9198961, -51.6881256, 51.3490601
18: -22.2069111, 16.3740158, -21.9954262, 16.3447895, -33.2000732, 33.1121979
19: -22.2729568, 6.0909448, -22.1007996, 6.0717020, -22.1136513, 21.9496155
20: -27.9230537, 0.6750603, -27.7821121, 0.6351995, -22.4655190, 22.3434753
21: -26.4728050, 7.4634032, -26.2422104, 7.4168491, -29.0496216, 28.8502655
22: -29.5435181, 5.3288617, -29.4582863, 5.3080654, -27.5185394, 27.4500809
23: -17.9282761, 12.3377657, -17.7724991, 12.3179159, -24.1229706, 23.9981804
24: -16.5363197, 13.6860514, -16.4690266, 13.6849432, -25.8962936, 25.8451614
25: -23.8474407, 8.9220209, -23.7805462, 8.9217615, -27.9415588, 27.9166870
26: -39.5111542, 4.1331034, -39.2480164, 4.0513525, -33.3983078, 33.1868134
27: -19.5899601, 14.9452419, -19.4883785, 14.9351282, -34.5250893, 34.4336205
28: -21.4802437, 11.5200825, -21.3272705, 11.4982624, -26.4047089, 26.2742500
29: -24.4967461, 6.4345994, -24.3686600, 6.4122829, -28.6830902, 28.5671005
30: -30.1397266, 5.9300294, -29.9707565, 5.8975906, -34.1312408, 33.9855347
31: -23.4591370, 7.7143135, -23.2987518, 7.7077436, -25.2044449, 25.0368347
32: -37.0088348, -2.5845881, -36.9847641, -2.6266050, -29.8166656, 29.8473053
33: -54.4603691, 0.2147779, -54.4222107, 0.0531483, -44.3783569, 44.6137924
34: -49.1624413, -6.9343662, -49.1332779, -6.9908004, -35.9011536, 35.8816986
35: -40.2979774, 3.9681606, -40.2886658, 3.9213562, -35.9958344, 36.0814896
36: -45.6134758, 1.3808899, -45.6004562, 1.3536015, -39.7331390, 39.7529526
37: -60.7362518, -9.8473339, -60.6842308, -9.8641186, -44.8357544, 44.9289627
38: -53.6765366, 4.1310787, -53.6406136, 4.0852556, -48.7831573, 48.9669113
39: -61.9331856, -4.4722109, -61.9340210, -4.5640631, -48.2693176, 48.4343109
40: -50.2062378, -9.1350727, -50.1654510, -9.1770630, -39.0867157, 39.1539230
41: -32.0196915, 7.3261032, -31.9934654, 7.3092599, -38.0221558, 38.0151978
42: -30.1119480, -0.2815547, -30.0989914, -0.3076148, -23.1178131, 23.1171761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1379
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7574690
time: 28.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7660530
time: 24.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1532326, 25.9984283, -8.0602760, 25.9204693, -30.9570389, 30.9527893
1: -0.4314270, 26.8769093, -0.3705101, 26.7814789, -21.4546471, 21.4495087
2: -0.4736919, 25.9471474, -0.3974762, 25.8051662, -20.9347534, 20.9624100
3: -4.8112073, 22.6464329, -4.7300110, 22.4810905, -19.9667244, 19.9740448
4: -7.7365646, 22.6384430, -7.6595030, 22.5092468, -25.7010956, 25.7058220
5: -4.9096632, 24.9668064, -4.8179731, 24.8139153, -23.8297272, 23.8573036
6: -39.4517441, -4.1195421, -39.4192619, -4.1474237, -29.8328094, 29.8273697
7: -9.2750816, 23.4785690, -9.1885290, 23.4188652, -25.8243179, 25.7515869
8: -13.9420509, 20.0963516, -13.8760290, 19.9796944, -29.3293571, 29.3188171
9: -8.6387072, 22.5511322, -8.6110725, 22.5094986, -27.9189835, 27.8001938
10: -29.1039696, 17.9489632, -28.9931831, 17.8637886, -39.7840195, 39.6736908
11: -26.4665337, 6.9224544, -26.1697121, 6.8670325, -29.9378357, 29.6772118
12: -46.2151222, -8.2544384, -46.0269470, -8.3928165, -32.4550858, 32.3619041
13: -32.6096649, 13.3287754, -32.6058884, 13.2290115, -38.6390228, 38.6394424
14: -59.7162209, -2.0453682, -59.4332352, -2.1518803, -57.1248779, 56.9024353
15: -14.2663021, 18.8745995, -14.2234821, 18.8115730, -28.0935516, 28.0709381
16: -15.7576389, 22.2969685, -15.6610012, 22.2781258, -29.7266617, 29.5517006
17: -59.4680290, -6.8507414, -59.1620750, -6.9175425, -51.7110443, 51.3986816
18: -22.2286682, 16.3799858, -22.0004234, 16.3469601, -33.2234268, 33.1193466
19: -22.3181839, 6.1117992, -22.1224403, 6.0721426, -22.1263466, 21.9913864
20: -27.9625931, 0.6922889, -27.8014183, 0.6359630, -22.4864273, 22.3843727
21: -26.5109596, 7.4828444, -26.2598152, 7.4183321, -29.0609283, 28.8851471
22: -29.5781269, 5.3472905, -29.4744034, 5.3102126, -27.5250931, 27.4812965
23: -17.9709892, 12.3648987, -17.7931023, 12.3205128, -24.1486664, 24.0500908
24: -16.5733528, 13.7021236, -16.4870243, 13.6859045, -25.9213791, 25.8826294
25: -23.9183464, 8.9595127, -23.8157196, 8.9255304, -27.9787674, 27.9941483
26: -39.5420456, 4.1518855, -39.2624359, 4.0546308, -33.4155579, 33.2227631
27: -19.5989990, 14.9504585, -19.4923687, 14.9367065, -34.5357056, 34.4428253
28: -21.5266800, 11.5465145, -21.3500671, 11.5002651, -26.4379044, 26.3274994
29: -24.5216217, 6.4479122, -24.3792915, 6.4138417, -28.6955566, 28.5856705
30: -30.1729565, 5.9534645, -29.9865704, 5.9000554, -34.1604462, 34.0265884
31: -23.5036507, 7.7399311, -23.3195992, 7.7088056, -25.2192841, 25.0864410
32: -37.0185928, -2.5735688, -36.9874954, -2.6228337, -29.8318176, 29.8603134
33: -54.4858360, 0.2344818, -54.4345894, 0.0554714, -44.3831329, 44.6440353
34: -49.1718483, -6.9226279, -49.1350250, -6.9868588, -35.9116898, 35.8965302
35: -40.3141785, 3.9774895, -40.2964020, 3.9227724, -35.9968872, 36.1042709
36: -45.6444092, 1.4017153, -45.6159592, 1.3552637, -39.7400208, 39.7772369
37: -60.7791977, -9.8247147, -60.7047653, -9.8620911, -44.8472137, 44.9616852
38: -53.7051849, 4.1468143, -53.6537476, 4.0881844, -48.7978821, 48.9884949
39: -61.9741974, -4.4449272, -61.9543533, -4.5609922, -48.2806396, 48.4726410
40: -50.2206459, -9.1251202, -50.1685753, -9.1744785, -39.1067047, 39.1631622
41: -32.0338020, 7.3491826, -31.9962997, 7.3197703, -38.0470123, 38.0379562
42: -30.1308689, -0.2629642, -30.1080551, -0.3033061, -23.1373024, 23.1439972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7784349
time: 32.15 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7871039
time: 29.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.1718445, 25.9767876, -8.1759100, 25.9293480, -30.9872093, 31.0329285
1: -0.4304180, 26.8444901, -0.4361572, 26.7900639, -21.4646873, 21.5039749
2: -0.5077386, 25.9205971, -0.5153451, 25.8314590, -20.9936218, 21.0398407
3: -4.8505344, 22.6118603, -4.8625026, 22.5132217, -20.0441246, 20.0387421
4: -7.7530136, 22.5945034, -7.7681437, 22.5287590, -25.7302208, 25.7469330
5: -4.9519830, 24.9387798, -4.9616394, 24.8482571, -23.9071884, 23.9284515
6: -39.4477158, -4.1219435, -39.4368095, -4.1284475, -29.8432236, 29.8498459
7: -9.2858944, 23.4501801, -9.2837629, 23.4193459, -25.8567619, 25.8455811
8: -13.9587908, 20.0672379, -13.9698391, 20.0085526, -29.3791199, 29.4044037
9: -8.6293478, 22.5466690, -8.6356497, 22.5443897, -27.9405441, 27.8692703
10: -29.0917835, 17.9816742, -29.0405998, 17.9684677, -39.8553162, 39.7639923
11: -26.4380417, 6.9471669, -26.2367134, 6.9566526, -29.9154205, 29.7761497
12: -46.2042847, -8.1629219, -46.0962753, -8.1645393, -32.6152344, 32.5322342
13: -32.5930367, 13.3270187, -32.6132126, 13.2665319, -38.6586761, 38.7010040
14: -59.7098694, -1.9550819, -59.5373001, -1.9590321, -57.2352295, 57.0973816
15: -14.2619190, 18.8451824, -14.2717400, 18.8186855, -28.0965805, 28.0686569
16: -15.7506027, 22.2837429, -15.7049217, 22.2879238, -29.7152863, 29.6072540
17: -59.4253616, -6.8245516, -59.2324028, -6.8066893, -51.7679749, 51.4904022
18: -22.2121716, 16.4079895, -22.0613537, 16.4106102, -33.2293091, 33.1620789
19: -22.2822762, 6.1253042, -22.1623039, 6.1405549, -22.1457748, 22.0513153
20: -27.9306431, 0.7221479, -27.8382759, 0.7305617, -22.5059738, 22.4593353
21: -26.4814930, 7.5186434, -26.3199444, 7.5280976, -29.1035690, 28.9958344
22: -29.5502377, 5.3469415, -29.4838276, 5.3493481, -27.5705490, 27.5600433
23: -17.9360867, 12.3704920, -17.8283634, 12.3852940, -24.1545639, 24.0720367
24: -16.5426769, 13.6946430, -16.4878159, 13.7034245, -25.9328003, 25.8651047
25: -23.8515816, 8.9461088, -23.8085060, 8.9720364, -27.9667435, 27.9598923
26: -39.5163383, 4.2285271, -39.3346481, 4.2400584, -33.5046921, 33.3687592
27: -19.6029034, 14.9540443, -19.5230713, 14.9520359, -34.5549393, 34.4771156
28: -21.4912567, 11.5570946, -21.3838062, 11.5736561, -26.4447708, 26.3644943
29: -24.5037079, 6.4553804, -24.4082527, 6.4575682, -28.7396088, 28.6746254
30: -30.1433525, 5.9602957, -30.0115166, 5.9635997, -34.1663284, 34.0503998
31: -23.4704609, 7.7393923, -23.3597641, 7.7588415, -25.2257080, 25.1100197
32: -37.0135422, -2.5540442, -37.0070686, -2.5614958, -29.8774719, 29.9059753
33: -54.5028076, 0.2262802, -54.5097771, 0.1191263, -44.4858093, 44.6489563
34: -49.1807556, -6.9285464, -49.1741333, -6.9543085, -35.9435577, 35.9082947
35: -40.3118248, 3.9700546, -40.3184929, 3.9405289, -36.0322876, 36.0904312
36: -45.6183052, 1.3938227, -45.6158752, 1.3883533, -39.7843628, 39.8136139
37: -60.7430344, -9.8234406, -60.7190018, -9.8122044, -44.9012299, 44.9795227
38: -53.6829834, 4.1481800, -53.6707230, 4.1290083, -48.8474731, 48.9968719
39: -61.9416809, -4.4576445, -61.9554443, -4.5170603, -48.3319244, 48.4739304
40: -50.2161331, -9.1297636, -50.1982231, -9.1560144, -39.1444550, 39.1834183
41: -32.0287743, 7.3356318, -32.0203590, 7.3319526, -38.0574646, 38.0524445
42: -30.1171074, -0.2676105, -30.1140480, -0.2742763, -23.1450806, 23.1534424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1379
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7768083
time: 29.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7768083
time: 27.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.2099276, 25.9993553, -8.1814852, 25.9404793, -31.0361404, 31.0528717
1: -0.4625292, 26.8780174, -0.4386683, 26.8065205, -21.5153885, 21.5166512
2: -0.5328269, 25.9491062, -0.5184464, 25.8456535, -21.0347595, 21.0531387
3: -4.8774686, 22.6484451, -4.8644753, 22.5316315, -20.0911331, 20.0476036
4: -7.7905998, 22.6413479, -7.7720833, 22.5519886, -25.7950935, 25.7762527
5: -4.9826193, 24.9705868, -4.9667444, 24.8643208, -23.9568367, 23.9485245
6: -39.4581642, -4.1109180, -39.4392128, -4.1245508, -29.8602295, 29.8604813
7: -9.3204679, 23.4809914, -9.2895031, 23.4345627, -25.9080811, 25.8600006
8: -13.9877768, 20.1004143, -13.9715233, 20.0240746, -29.4216995, 29.4114647
9: -8.6479197, 22.5724258, -8.6383104, 22.5573006, -27.9747772, 27.8863068
10: -29.1099777, 18.0024395, -29.0431023, 17.9781532, -39.8890839, 39.7744293
11: -26.4702530, 6.9683113, -26.2496395, 6.9581933, -29.9355011, 29.8123550
12: -46.2182083, -8.1396027, -46.1023407, -8.1594028, -32.6299896, 32.5508995
13: -32.6188812, 13.3442039, -32.6250763, 13.2697344, -38.6783600, 38.7149048
14: -59.7351685, -1.9489326, -59.5474167, -1.9574652, -57.2562866, 57.1119690
15: -14.2888393, 18.8770180, -14.2747879, 18.8338814, -28.1409378, 28.0942345
16: -15.7717190, 22.3062248, -15.7087402, 22.2999001, -29.7516785, 29.6111259
17: -59.4752846, -6.7962627, -59.2522163, -6.8043613, -51.7909698, 51.5400543
18: -22.2340050, 16.4140186, -22.0663223, 16.4127464, -33.2526779, 33.1692505
19: -22.3275223, 6.1461945, -22.1839638, 6.1409931, -22.1584778, 22.0931091
20: -27.9701328, 0.7393851, -27.8576221, 0.7313571, -22.5268631, 22.5002251
21: -26.5196285, 7.5380745, -26.3375492, 7.5296483, -29.1149063, 29.0306892
22: -29.5848122, 5.3653712, -29.4999161, 5.3514819, -27.5771332, 27.5912552
23: -17.9787827, 12.3976040, -17.8489590, 12.3879147, -24.1802673, 24.1239433
24: -16.5797005, 13.7107315, -16.5058327, 13.7043772, -25.9579315, 25.9025574
25: -23.9224796, 8.9836016, -23.8436775, 8.9758701, -28.0039673, 28.0373840
26: -39.5472374, 4.2472601, -39.3491325, 4.2432890, -33.5219650, 33.4046860
27: -19.6119442, 14.9591837, -19.5270538, 14.9536419, -34.5655861, 34.4862366
28: -21.5376740, 11.5835438, -21.4066200, 11.5756435, -26.4779968, 26.4177628
29: -24.5286255, 6.4686847, -24.4189167, 6.4591265, -28.7521057, 28.6932220
30: -30.1765938, 5.9837632, -30.0272923, 5.9660344, -34.1955185, 34.0914612
31: -23.5149574, 7.7650433, -23.3805904, 7.7598538, -25.2406158, 25.1596336
32: -37.0232964, -2.5430527, -37.0098038, -2.5577159, -29.8926620, 29.9189606
33: -54.5283356, 0.2460003, -54.5222015, 0.1215296, -44.4905853, 44.6792526
34: -49.1901932, -6.9167681, -49.1758690, -6.9503860, -35.9541321, 35.9231491
35: -40.3280525, 3.9793606, -40.3262978, 3.9420166, -36.0333328, 36.1132584
36: -45.6492844, 1.4147072, -45.6313057, 1.3899784, -39.7913742, 39.8378906
37: -60.7860184, -9.8007946, -60.7395210, -9.8101454, -44.9127502, 45.0122910
38: -53.7116241, 4.1639185, -53.6838989, 4.1320324, -48.8621521, 49.0184631
39: -61.9827232, -4.4302759, -61.9757423, -4.5140076, -48.3433380, 48.5121689
40: -50.2305222, -9.1197977, -50.2013435, -9.1534338, -39.1645355, 39.1927490
41: -32.0428925, 7.3586946, -32.0231895, 7.3425064, -38.0823669, 38.0753326
42: -30.1360168, -0.2490625, -30.1231651, -0.2699337, -23.1645813, 23.1802292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=189, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7977740
time: 34.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -13.7585241, upper bound: 13.8064511
time: 32.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 68.96 seconds
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.6997229, upper bound: 13.7970446
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.6997229, upper bound: 13.8058849
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7223054, upper bound: 13.7977740
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7223054, upper bound: 13.8064511
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7359460, upper bound: 13.7970446
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7359460, upper bound: 13.8058849
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7574690
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7660530
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7784349
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7871039
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7768083
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7768083
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.7977740
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 68.96
Output dim: 1, lower bound: -13.7585241, upper bound: 13.8064511

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9627523, 25.9145584, -8.0784140, 25.9377270, -30.8019562, 30.8604622
1: -0.3018036, 26.7840958, -0.3731503, 26.8046684, -21.3516541, 21.3552704
2: -0.3543468, 25.8189220, -0.4416504, 25.8425026, -20.8545609, 20.8467636
3: -4.6994920, 22.5158348, -4.7878528, 22.5288830, -19.9307480, 19.8788528
4: -7.5961528, 22.5297356, -7.6885662, 22.5480995, -25.5952148, 25.5768776
5: -4.7854004, 24.8379211, -4.8819427, 24.8603916, -23.7703056, 23.7384720
6: -39.4154739, -4.2035675, -39.4306183, -4.1598501, -29.7717361, 29.7565384
7: -9.1247931, 23.4003525, -9.2090759, 23.4317665, -25.6974678, 25.7000198
8: -13.8289433, 19.9920654, -13.9014864, 20.0175686, -29.2440643, 29.2297440
9: -8.5272799, 22.5193386, -8.5859547, 22.5528679, -27.7961578, 27.7984390
10: -28.9532528, 17.8826485, -28.9979019, 17.9481773, -39.6712646, 39.6100540
11: -26.2012291, 6.8354826, -26.2270336, 6.9023013, -29.6094208, 29.6916275
12: -46.0782776, -8.3817348, -46.0969391, -8.2620153, -32.3953400, 32.3315849
13: -32.5328636, 13.2303476, -32.5925789, 13.2540169, -38.5710373, 38.5709610
14: -59.4314346, -2.1362343, -59.4925003, -2.0267220, -56.8919678, 56.9225922
15: -14.2133350, 18.8103065, -14.2565861, 18.8221283, -28.0355377, 28.0182838
16: -15.5855179, 22.2522278, -15.6500874, 22.3053303, -29.5439987, 29.4965591
17: -59.1707306, -6.9322243, -59.2130699, -6.8580093, -51.4208450, 51.4184113
18: -22.0275726, 16.2938728, -22.0543900, 16.3598938, -33.0639114, 33.0688477
19: -22.1473236, 6.0224133, -22.1715927, 6.0866013, -21.9181099, 21.9668732
20: -27.8213482, 0.5993681, -27.8489170, 0.6703939, -22.3079147, 22.3736458
21: -26.2959614, 7.3831863, -26.3221436, 7.4628539, -28.8175354, 28.8752937
22: -29.4471931, 5.2552042, -29.4877815, 5.3042498, -27.3843155, 27.4514542
23: -17.8160839, 12.2661686, -17.8393593, 12.3326826, -23.9577408, 23.9897537
24: -16.4766598, 13.6148853, -16.4983711, 13.6623878, -25.8332977, 25.8085785
25: -23.8082924, 8.8623676, -23.8373642, 8.9227057, -27.8387375, 27.9037361
26: -39.2975655, 4.0020237, -39.3401527, 4.1330433, -33.1543350, 33.1808167
27: -19.4778347, 14.8497286, -19.5091019, 14.9044209, -34.3822556, 34.3588295
28: -21.3623009, 11.4258623, -21.3947811, 11.5060024, -26.2284775, 26.2555771
29: -24.3709221, 6.3701587, -24.4045906, 6.4166608, -28.5281982, 28.5617599
30: -30.0046272, 5.8490705, -30.0187416, 5.9140821, -33.9654083, 33.9642029
31: -23.3387909, 7.6443462, -23.3664589, 7.7066989, -25.0040207, 25.0406723
32: -36.9845810, -2.6319456, -37.0017014, -2.5911331, -29.8198700, 29.8197021
33: -54.4036331, 0.0278511, -54.4831543, 0.0751209, -44.3424530, 44.3899994
34: -49.1195946, -7.0499139, -49.1679764, -6.9993992, -35.8019104, 35.7904053
35: -40.2657814, 3.8550653, -40.3200607, 3.8977261, -35.9143066, 35.9583893
36: -45.5706291, 1.2821608, -45.6235199, 1.3357258, -39.6270599, 39.6872787
37: -60.6870499, -9.9111338, -60.7308273, -9.8552656, -44.7923126, 44.8526688
38: -53.6025734, 3.9910412, -53.6738396, 4.0638542, -48.7081451, 48.7692566
39: -61.9026031, -4.5962934, -61.9527969, -4.5542622, -48.2269592, 48.3104324
40: -50.1669388, -9.1962566, -50.1886063, -9.1723537, -39.0888977, 39.0781326
41: -31.9797478, 7.2651796, -32.0125732, 7.3078828, -37.9834290, 37.9709854
42: -30.0971470, -0.3426790, -30.1168690, -0.3028250, -23.0893173, 23.0748062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6989717, upper bound: 13.7481812
time: 24.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.6989898, upper bound: 13.7699963
time: 31.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.0890131, 25.9379120, -8.1414871, 25.9396133, -30.9290428, 30.9477348
1: -0.3855519, 26.8030319, -0.4150457, 26.8054352, -21.4230080, 21.4169655
2: -0.4227424, 25.8403397, -0.4764404, 25.8435631, -20.9067612, 20.9026680
3: -4.7542515, 22.5300179, -4.8159571, 22.5310173, -19.9730377, 19.9260368
4: -7.6909695, 22.5447159, -7.7359791, 22.5494633, -25.6704140, 25.6424789
5: -4.8543906, 24.8604355, -4.9174342, 24.8629303, -23.8235779, 23.7988319
6: -39.4284210, -4.1327076, -39.4346924, -4.1249771, -29.8387833, 29.8330536
7: -9.2257366, 23.4306564, -9.2605419, 23.4333496, -25.7697220, 25.7841301
8: -13.9195356, 20.0146637, -13.9480515, 20.0205059, -29.3203621, 29.2998619
9: -8.6208782, 22.5504494, -8.6318941, 22.5543098, -27.8710632, 27.8765106
10: -29.0348663, 17.9150467, -29.0394535, 17.9528542, -39.7320404, 39.6825714
11: -26.2403641, 6.8546395, -26.2463703, 6.9110966, -29.6663857, 29.7303123
12: -46.0971985, -8.2984600, -46.1003799, -8.2209482, -32.4512405, 32.4267044
13: -32.5843124, 13.2531261, -32.6181755, 13.2620716, -38.6339874, 38.6498795
14: -59.5180244, -2.1058073, -59.5348091, -2.0228863, -56.9876709, 57.0231934
15: -14.2544880, 18.8284607, -14.2750568, 18.8321934, -28.0883484, 28.0617561
16: -15.6840076, 22.2929363, -15.6981735, 22.3064384, -29.6030273, 29.5876007
17: -59.2369308, -6.9087162, -59.2461281, -6.8495884, -51.4974899, 51.5361023
18: -22.0573196, 16.3309460, -22.0628738, 16.3783073, -33.1135101, 33.1110382
19: -22.1695995, 6.0565019, -22.1779251, 6.1040211, -21.9600830, 22.0025444
20: -27.8447342, 0.6443825, -27.8525085, 0.6929455, -22.3542099, 22.4153366
21: -26.3211937, 7.4167562, -26.3308697, 7.4797959, -28.8618011, 28.9151154
22: -29.4857903, 5.3112822, -29.4946480, 5.3333325, -27.4531555, 27.4850807
23: -17.8362961, 12.3072548, -17.8437767, 12.3528805, -24.0000381, 24.0226974
24: -16.4966125, 13.6489601, -16.5026226, 13.6795311, -25.8704071, 25.8383560
25: -23.8348694, 8.9149456, -23.8407555, 8.9496031, -27.8934021, 27.9347839
26: -39.3394432, 4.0787096, -39.3453865, 4.1719136, -33.2353516, 33.2391052
27: -19.5073261, 14.8899632, -19.5190239, 14.9244490, -34.4317741, 34.4089890
28: -21.3885250, 11.4808731, -21.3990669, 11.5343809, -26.2858200, 26.3003845
29: -24.4022064, 6.4137955, -24.4131546, 6.4392424, -28.5838013, 28.5954819
30: -30.0168343, 5.8710036, -30.0237083, 5.9240408, -33.9939728, 33.9952469
31: -23.3642654, 7.6873693, -23.3735447, 7.7282438, -25.0543976, 25.0917931
32: -36.9995613, -2.5790625, -37.0059586, -2.5648408, -29.8744202, 29.8790054
33: -54.4448051, 0.1032629, -54.4890289, 0.1129551, -44.4257965, 44.4329453
34: -49.1640778, -6.9592276, -49.1713104, -6.9542809, -35.8919525, 35.8641663
35: -40.3115883, 3.9398308, -40.3245010, 3.9406796, -36.0024948, 36.0150452
36: -45.6228981, 1.3797884, -45.6282349, 1.3848591, -39.7290649, 39.7740326
37: -60.7322083, -9.8401690, -60.7368507, -9.8188305, -44.8752441, 44.8904190
38: -53.6734314, 4.1219835, -53.6805649, 4.1287098, -48.8462372, 48.8843613
39: -61.9448433, -4.5355930, -61.9631348, -4.5240412, -48.3009338, 48.3702240
40: -50.1904373, -9.1641188, -50.1971245, -9.1567860, -39.1336975, 39.1218567
41: -32.0090714, 7.3286872, -32.0173721, 7.3395920, -38.0520020, 38.0393906
42: -30.1143131, -0.2830415, -30.1194038, -0.2730246, -23.1317902, 23.1351547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7215553, upper bound: 13.7364995
time: 30.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7215737, upper bound: 13.7705127
time: 31.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.0801668, 25.9758587, -8.1184635, 25.9385872, -30.9054260, 30.9663620
1: -0.3763957, 26.8590069, -0.3968201, 26.8057747, -21.4191589, 21.4548912
2: -0.4623508, 25.9276009, -0.4836369, 25.8446045, -20.9512329, 20.9971657
3: -4.8208656, 22.6340618, -4.8363881, 22.5295162, -20.0115662, 20.0002327
4: -7.6927786, 22.6262703, -7.7246590, 22.5506210, -25.6803055, 25.7104721
5: -4.9114532, 24.9478035, -4.9312105, 24.8617859, -23.8678780, 23.8878479
6: -39.4449844, -4.1828136, -39.4350815, -4.1593876, -29.8105087, 29.7818909
7: -9.2166185, 23.4505424, -9.2379856, 23.4329796, -25.7809525, 25.7757339
8: -13.8942461, 20.0776234, -13.9249048, 20.0211849, -29.3106918, 29.3411522
9: -8.5517521, 22.5411339, -8.5923414, 22.5558090, -27.8680420, 27.8081360
10: -29.0258942, 17.9696922, -29.0016594, 17.9734421, -39.7919769, 39.7016144
11: -26.4298058, 6.9482942, -26.2302780, 6.9493876, -29.8851166, 29.7726097
12: -46.1990242, -8.2245293, -46.0988464, -8.2004566, -32.5731583, 32.4587479
13: -32.5658951, 13.3208408, -32.5994644, 13.2617168, -38.6135254, 38.6565857
14: -59.6467018, -1.9795771, -59.5051498, -1.9612684, -57.1582794, 57.0404358
15: -14.2464066, 18.8583069, -14.2563190, 18.8238220, -28.0866394, 28.0516777
16: -15.6706486, 22.2654667, -15.6606369, 22.2987919, -29.6326294, 29.5198898
17: -59.4071198, -6.8212223, -59.2192230, -6.8128357, -51.7121277, 51.5004425
18: -22.2036419, 16.3757439, -22.0578098, 16.3943386, -33.2024460, 33.1162262
19: -22.3049202, 6.1106567, -22.1776543, 6.1235399, -22.1161880, 22.0368576
20: -27.9465294, 0.6929703, -27.8540173, 0.7087698, -22.4803238, 22.4480743
21: -26.4939880, 7.5031304, -26.3288651, 7.5126662, -29.0701294, 28.9767113
22: -29.5457878, 5.3073697, -29.4930553, 5.3224916, -27.5077896, 27.5059700
23: -17.9582405, 12.3550110, -17.8445511, 12.3676987, -24.1377029, 24.0609207
24: -16.5594177, 13.6752968, -16.5015907, 13.6872406, -25.9203796, 25.8545609
25: -23.8956528, 8.9289522, -23.8402863, 8.9489412, -27.9490738, 27.9546738
26: -39.5049744, 4.1683407, -39.3438568, 4.2043920, -33.4405899, 33.3115158
27: -19.5818253, 14.9175892, -19.5171127, 14.9335947, -34.5154190, 34.4347000
28: -21.5111408, 11.5264874, -21.4023209, 11.5472622, -26.4203186, 26.3378296
29: -24.4968071, 6.4234262, -24.4103298, 6.4365301, -28.6959457, 28.6184921
30: -30.1639729, 5.9605761, -30.0223389, 5.9561014, -34.1658936, 34.0582886
31: -23.4891376, 7.7202802, -23.3734913, 7.7383451, -25.1898880, 25.0979614
32: -37.0079308, -2.5972991, -37.0055733, -2.5839987, -29.8562317, 29.8583679
33: -54.4866791, 0.1685238, -54.5163345, 0.0837097, -44.4066162, 44.5773544
34: -49.1452980, -7.0096283, -49.1725807, -6.9955111, -35.8637543, 35.8194885
35: -40.2818222, 3.8920994, -40.3219261, 3.8990879, -35.9447632, 36.0052490
36: -45.5965996, 1.3143625, -45.6265831, 1.3408337, -39.6890182, 39.7244339
37: -60.7403793, -9.8730135, -60.7334900, -9.8465080, -44.8291168, 44.9194641
38: -53.6403503, 4.0295477, -53.6772652, 4.0672264, -48.7235260, 48.8628845
39: -61.9397469, -4.4919281, -61.9654694, -4.5441990, -48.2686005, 48.4335327
40: -50.2057419, -9.1528893, -50.1928406, -9.1689720, -39.1173553, 39.1472549
41: -32.0132408, 7.2940598, -32.0183372, 7.3108134, -38.0213470, 38.0056686
42: -30.1186829, -0.3095770, -30.1205502, -0.2997317, -23.1219025, 23.1089706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7351942, upper bound: 13.7481810
time: 29.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7352130, upper bound: 13.7699961
time: 27.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.2063713, 25.9992447, -8.1814852, 25.9404793, -31.0325241, 31.0536652
1: -0.4601736, 26.8779564, -0.4386683, 26.8065205, -21.4905014, 21.5165863
2: -0.5307264, 25.9490395, -0.5184464, 25.8456535, -21.0034828, 21.0530396
3: -4.8756266, 22.6482849, -4.8644753, 22.5316315, -20.0538826, 20.0473976
4: -7.7876530, 22.6412430, -7.7720833, 22.5519886, -25.7555008, 25.7760582
5: -4.9804392, 24.9703636, -4.9667444, 24.8643208, -23.9211731, 23.9482689
6: -39.4578857, -4.1119366, -39.4392128, -4.1245508, -29.8775101, 29.8583984
7: -9.3175478, 23.4808655, -9.2895031, 23.4345627, -25.8532104, 25.8598328
8: -13.9848385, 20.1002140, -13.9715233, 20.0240746, -29.3869896, 29.4112778
9: -8.6453438, 22.5722752, -8.6383104, 22.5573006, -27.9428558, 27.8862000
10: -29.1074696, 18.0021095, -29.0431023, 17.9781532, -39.8527527, 39.7740936
11: -26.4689217, 6.9674344, -26.2496395, 6.9581933, -29.9420776, 29.8112717
12: -46.2179565, -8.1412773, -46.1023407, -8.1594028, -32.6290588, 32.5538483
13: -32.6173325, 13.3436890, -32.6250763, 13.2697344, -38.6765137, 38.7355347
14: -59.7330894, -1.9491825, -59.5474167, -1.9574652, -57.2539673, 57.1410065
15: -14.2875652, 18.8764534, -14.2747879, 18.8338814, -28.1394272, 28.0951653
16: -15.7691231, 22.3061523, -15.7087402, 22.2999001, -29.6917038, 29.6109886
17: -59.4734154, -6.7977743, -59.2522163, -6.8043613, -51.7888184, 51.6180420
18: -22.2334118, 16.4128761, -22.0663223, 16.4127464, -33.2521057, 33.1583481
19: -22.3271637, 6.1447668, -22.1839638, 6.1409931, -22.1581573, 22.0725327
20: -27.9698868, 0.7380147, -27.8576221, 0.7313571, -22.5266037, 22.4897385
21: -26.5192013, 7.5367017, -26.3375492, 7.5296483, -29.1143417, 29.0165367
22: -29.5844078, 5.3634257, -29.4999161, 5.3514819, -27.5766220, 27.5396156
23: -17.9784927, 12.3960648, -17.8489590, 12.3879147, -24.1799698, 24.0938950
24: -16.5793419, 13.7093811, -16.5058327, 13.7043772, -25.9575806, 25.8844070
25: -23.9222260, 8.9815426, -23.8436775, 8.9758701, -28.0036469, 27.9857178
26: -39.5468712, 4.2450762, -39.3491325, 4.2432890, -33.5215988, 33.3697739
27: -19.6112957, 14.9578753, -19.5270538, 14.9536419, -34.5649376, 34.4849281
28: -21.5373650, 11.5814762, -21.4066200, 11.5756435, -26.4776382, 26.3826408
29: -24.5281258, 6.4670601, -24.4189167, 6.4591265, -28.7515411, 28.6522255
30: -30.1762104, 5.9825521, -30.0272923, 5.9660344, -34.1944656, 34.0893097
31: -23.5146008, 7.7633228, -23.3805904, 7.7598538, -25.2402573, 25.1490593
32: -37.0229568, -2.5444098, -37.0098038, -2.5577159, -29.9108047, 29.9176407
33: -54.5278854, 0.2437792, -54.5222015, 0.1215296, -44.4900665, 44.6201859
34: -49.1898346, -6.9189653, -49.1758690, -6.9503860, -35.9537659, 35.8932190
35: -40.3276291, 3.9769087, -40.3262978, 3.9420166, -36.0329514, 36.0619583
36: -45.6488876, 1.4119997, -45.6313057, 1.3899784, -39.7910385, 39.8111725
37: -60.7855606, -9.8020439, -60.7395210, -9.8101454, -44.9121399, 44.9572296
38: -53.7112007, 4.1604462, -53.6838989, 4.1320324, -48.8617401, 48.9780807
39: -61.9819832, -4.4313412, -61.9757423, -4.5140076, -48.3425751, 48.4932861
40: -50.2292557, -9.1208067, -50.2013435, -9.1534338, -39.1621857, 39.1910095
41: -32.0425644, 7.3575807, -32.0231895, 7.3425064, -38.0899353, 38.0740356
42: -30.1358624, -0.2499623, -30.1231651, -0.2699337, -23.1643524, 23.1693268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=188, inp2_unstable=189, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1716

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7577745, upper bound: 13.7364995
time: 29.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -13.7577923, upper bound: 13.7705125
time: 31.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 63.90 seconds
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.6989717, upper bound: 13.7481812
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.6989898, upper bound: 13.7699963
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7215553, upper bound: 13.7364995
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7215737, upper bound: 13.7705127
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7351942, upper bound: 13.7481810
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7352130, upper bound: 13.7699961
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7577745, upper bound: 13.7364995
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 63.90
Output dim: 1, lower bound: -13.7577923, upper bound: 13.7705125

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 53.24 + 2080.19 = 2133.42 seconds
