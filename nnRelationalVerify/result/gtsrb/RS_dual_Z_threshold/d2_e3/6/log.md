## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 6)
Time budget: 3600 seconds
Split limit: 100
Threshold: 13.5190456218


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8179855, 26.8179855)
1: (-10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5941849, 13.5941849)
2: (-14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3140030, 14.3140030)
3: (-21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6006126, 19.6006088)
4: (-22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5998955, 19.5998993)
5: (-20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1066132, 23.1066132)
6: (-22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2610626, 21.2610588)
7: (-21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1534576, 21.1534576)
8: (-34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8828506, 20.8828545)
9: (-12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4273911, 26.4273911)
10: (-6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7357330, 23.7357330)
11: (-6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5698509, 18.5698509)
12: (0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.7007065, 28.7007065)
13: (-10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250763, 30.1250763)
14: (-33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0085297, 38.0085297)
15: (-20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6258430, 18.6258430)
16: (-14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293)
17: (-21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7759247, 36.7759323)
18: (-14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1603165, 21.1603127)
19: (-10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9675522, 14.9675522)
20: (-15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9665451, 17.9665375)
21: (-11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6507721, 18.6507721)
22: (-9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4127159, 15.4127159)
23: (-14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5985527, 19.5985489)
24: (-17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7441673, 18.7441711)
25: (-11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4965668, 20.4965706)
26: (-16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7661095, 24.7661133)
27: (-27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3572273, 20.3572235)
28: (-16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8372993, 20.8372993)
29: (-7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1900558, 16.1900597)
30: (-19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8333130, 21.8333130)
31: (-13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0579376, 19.0579338)
32: (-12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1630745, 18.1630707)
33: (-45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3682098, 31.3682175)
34: (-42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8698883, 19.8698921)
35: (-29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8463821, 21.8463821)
36: (-23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5975952, 23.5975914)
37: (-43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2220230, 36.2220230)
38: (-30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3143082, 29.3143082)
39: (-38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4811401, 32.4811478)
40: (-44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2817078, 26.2817078)
41: (-24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3723526, 23.3723526)
42: (-19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6306076, 16.6306076)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.42 + 47.02 = 49.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -13.5325782, upper bound: 13.5325782

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5014864, upper bound: 13.5315658
time: 41.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5315658, upper bound: 13.5014864
time: 23.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 65.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 65.31
Output dim: 12, lower bound: -13.5014864, upper bound: 13.5315658
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 65.31
Output dim: 12, lower bound: -13.5315658, upper bound: 13.5014864

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8256073, 26.8244400
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6025124, 13.6012859
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3114662, 14.3089790
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5978928, 19.5979919
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6358490, 19.6291199
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1068344, 23.1067810
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2278366, 21.2315750
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1557236, 21.1542549
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.9047623, 20.8984718
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4238586, 26.4240875
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7338333, 23.7342339
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5800743, 18.5810165
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6934204, 28.6940994
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250076, 30.1250153
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9989166, 37.9970322
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6035538, 18.5985451
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7697906, 36.7696609
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1625671, 21.1618156
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9536667, 14.9517784
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9781418, 17.9776459
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6535416, 18.6534195
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4057922, 15.4049568
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6057968, 19.6060028
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7422218, 18.7420502
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5016403, 20.5012779
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7688751, 24.7684402
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3536720, 20.3533173
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8534012, 20.8539963
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1883430, 16.1882553
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8624077, 21.8636818
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0379219, 19.0344391
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1630020, 18.1630096
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3586578, 31.3587875
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8633957, 19.8639221
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8272476, 21.8273239
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5854454, 23.5866127
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2281647, 36.2276993
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3126221, 29.3128548
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5072861, 32.5032043
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2848434, 26.2846680
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3711853, 23.3713608
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6112061, 16.6132202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4930583, upper bound: 13.5303422
time: 43.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5003690, upper bound: 13.5230493
time: 25.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8244476, 26.8256073
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6012878, 13.6025085
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3089790, 14.3114662
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5979919, 19.5978928
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6291199, 19.6358490
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1067810, 23.1068344
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2315750, 21.2278366
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1542511, 21.1557236
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8984680, 20.9047623
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4240875, 26.4238586
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7342377, 23.7338333
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5810127, 18.5800705
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6940918, 28.6934280
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250153, 30.1250076
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9970398, 37.9989090
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5985489, 18.6035576
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7696686, 36.7697983
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1618195, 21.1625633
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9517784, 14.9536667
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9776459, 17.9781456
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6534195, 18.6535454
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4049568, 15.4057922
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6059952, 19.6058006
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7420540, 18.7422218
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5012741, 20.5016441
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7684402, 24.7688751
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3533211, 20.3536644
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8539963, 20.8534050
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1882515, 16.1883430
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8636818, 21.8624039
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0344429, 19.0379257
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1630096, 18.1630058
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3587875, 31.3586540
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8639221, 19.8633995
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8273239, 21.8272476
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5866203, 23.5854454
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2276917, 36.2281647
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3128510, 29.3126221
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5032043, 32.5072861
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2846680, 26.2848396
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3713608, 23.3711815
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6132202, 16.6112080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5230493, upper bound: 13.5003690
time: 37.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5303422, upper bound: 13.4930583
time: 25.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 64.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 64.44
Output dim: 12, lower bound: -13.4930583, upper bound: 13.5303422
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 64.44
Output dim: 12, lower bound: -13.5003690, upper bound: 13.5230493
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 64.44
Output dim: 12, lower bound: -13.5230493, upper bound: 13.5003690
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 64.44
Output dim: 12, lower bound: -13.5303422, upper bound: 13.4930583

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8133087, 26.8139114
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5967712, 13.5961781
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3093529, 14.3070030
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5842896, 19.5857658
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6371956, 19.6304932
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0905685, 23.0922852
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2151947, 21.2207985
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1432495, 21.1431503
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8984222, 20.8926926
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4168930, 26.4178925
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7342911, 23.7348633
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5806046, 18.5814972
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6979675, 28.6986237
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250992, 30.1251144
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9985504, 37.9971237
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5977325, 18.5923080
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7566986, 36.7577972
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1629295, 21.1621284
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9447174, 14.9417839
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9851952, 17.9847946
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6524506, 18.6522141
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4025459, 15.4013062
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6021347, 19.6018791
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7413597, 18.7411461
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5038681, 20.5033989
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7701492, 24.7696381
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3565483, 20.3559875
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8553276, 20.8558884
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1870155, 16.1868019
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8736572, 21.8747711
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0271492, 19.0221634
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1639709, 18.1641617
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3617401, 31.3611145
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8633156, 19.8636932
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8271637, 21.8264465
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5854149, 23.5865288
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2352829, 36.2327347
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3120346, 29.3122177
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5194626, 32.5130920
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2872696, 26.2868080
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3709106, 23.3706169
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6085663, 16.6113052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4923613, upper bound: 13.5216892
time: 40.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4843918, upper bound: 13.5296484
time: 24.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8150787, 26.8121490
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5974045, 13.5955486
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3094559, 14.3068657
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5857086, 19.5843849
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6372261, 19.6305008
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0923233, 23.0905151
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2168884, 21.2189369
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1445084, 21.1417770
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8989792, 20.8921356
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4176712, 26.4171219
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7344589, 23.7346382
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5805435, 18.5815506
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6979523, 28.6986160
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1251068, 30.1251068
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9990082, 37.9966660
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5973206, 18.5928192
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7578888, 36.7565613
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1628761, 21.1621780
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9436722, 14.9428291
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9851875, 17.9846954
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6523438, 18.6523209
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4021416, 15.4017105
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6016769, 19.6023026
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7413139, 18.7411842
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5037308, 20.5034981
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7700577, 24.7697220
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3563347, 20.3562012
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8552361, 20.8559151
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1868935, 16.1869240
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8733215, 21.8749313
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0256462, 19.0236740
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1641541, 18.1639786
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3609772, 31.3618736
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8631477, 19.8638382
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8263702, 21.8271751
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5853615, 23.5865860
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2332077, 36.2347641
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3120117, 29.3122635
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5171738, 32.5154953
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2869797, 26.2870560
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3704376, 23.3710785
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6092033, 16.6105785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4996734, upper bound: 13.5143874
time: 27.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4917076, upper bound: 13.5223534
time: 54.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8121490, 26.8150787
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5955467, 13.5974045
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3068657, 14.3094559
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5843887, 19.5857086
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6304970, 19.6372223
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0905151, 23.0923233
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2189331, 21.2168922
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1417770, 21.1445084
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8921356, 20.8989792
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4171219, 26.4176712
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7346344, 23.7344589
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5815506, 18.5805397
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6986084, 28.6979523
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1251068, 30.1251068
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9966736, 37.9990005
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5928192, 18.5973206
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7565613, 36.7578812
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1621819, 21.1628799
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9428291, 14.9436722
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9846916, 17.9851837
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6523209, 18.6523438
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4017105, 15.4021416
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6023026, 19.6016769
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7411842, 18.7413177
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5035019, 20.5037270
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7697144, 24.7700577
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3562050, 20.3563385
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8559074, 20.8552399
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1869240, 16.1868935
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8749237, 21.8733177
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0236702, 19.0256500
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1639786, 18.1641579
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3618698, 31.3609810
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8638420, 19.8631477
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8271790, 21.8263664
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5865822, 23.5853577
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2347641, 36.2332001
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3122635, 29.3120117
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5154953, 32.5171738
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2870560, 26.2869835
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3710785, 23.3704376
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6105766, 16.6092033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5223534, upper bound: 13.4917076
time: 23.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5143874, upper bound: 13.4996734
time: 46.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8139114, 26.8133087
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5961800, 13.5967712
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3070030, 14.3093567
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5857620, 19.5842896
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6304970, 19.6371994
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0922852, 23.0905685
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2207947, 21.2151985
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1431503, 21.1432495
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8926926, 20.8984261
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4178925, 26.4168930
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7348633, 23.7342911
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5814972, 18.5806084
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6986237, 28.6979752
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1251221, 30.1250992
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9971313, 37.9985504
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5923080, 18.5977287
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7577972, 36.7566986
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1621284, 21.1629295
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9417839, 14.9447174
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9847984, 17.9851952
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6522141, 18.6524544
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4013062, 15.4025459
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6018753, 19.6021347
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7411461, 18.7413559
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5033951, 20.5038643
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7696381, 24.7701492
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3559914, 20.3565483
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8558922, 20.8553200
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1868019, 16.1870155
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8747711, 21.8736534
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0221672, 19.0271492
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1641617, 18.1639748
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3611145, 31.3617401
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8636971, 19.8633118
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8264465, 21.8271599
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5865288, 23.5854111
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2327194, 36.2352982
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3122177, 29.3120308
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5130920, 32.5194626
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2868118, 26.2872734
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3706131, 23.3709106
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6113052, 16.6085663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5296484, upper bound: 13.4843918
time: 31.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5216892, upper bound: 13.4923613
time: 32.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 65.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.4923613, upper bound: 13.5216892
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.4843918, upper bound: 13.5296484
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.4996734, upper bound: 13.5143874
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.4917076, upper bound: 13.5223534
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.5223534, upper bound: 13.4917076
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.5143874, upper bound: 13.4996734
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.5296484, upper bound: 13.4843918
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.43
Output dim: 12, lower bound: -13.5216892, upper bound: 13.4923613

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8095169, 26.8082657
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5951042, 13.5938797
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3099518, 14.3066711
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5843887, 19.5857162
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6370468, 19.6302452
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0895386, 23.0914383
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2104416, 21.2173691
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1428986, 21.1427841
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8967896, 20.8902321
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4166870, 26.4175034
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7321167, 23.7329712
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5802345, 18.5812798
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6968079, 28.6996536
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1244812, 30.1250992
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9934998, 37.9940262
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5968895, 18.5910263
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7523041, 36.7563095
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1603928, 21.1586723
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9449081, 14.9414482
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9851761, 17.9847908
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6521606, 18.6520920
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4012756, 15.4011307
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6015778, 19.5992661
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7394600, 18.7389412
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5038376, 20.5033989
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7669373, 24.7647095
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3562851, 20.3551102
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8552017, 20.8557358
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1846809, 16.1855354
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8662415, 21.8694344
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0270195, 19.0220337
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1619873, 18.1621399
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3615952, 31.3609314
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8630295, 19.8633728
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8248863, 21.8234749
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5810547, 23.5805321
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2306061, 36.2257767
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3033676, 29.3005676
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5162964, 32.5086365
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2868195, 26.2861061
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3666229, 23.3647308
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6061554, 16.6079140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4698078, upper bound: 13.5210366
time: 19.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4917047, upper bound: 13.4993801
time: 27.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8076630, 26.8101196
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5944710, 13.5945110
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3090248, 14.3075981
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5842361, 19.5858688
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6369476, 19.6303482
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0897217, 23.0912628
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2117691, 21.2160416
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1428833, 21.1427994
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8959656, 20.8910599
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4165039, 26.4176865
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7323990, 23.7326851
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5803871, 18.5811234
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6990051, 28.6974564
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250839, 30.1245003
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9954529, 37.9920654
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5964470, 18.5914650
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7552032, 36.7533951
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1594696, 21.1595917
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9443817, 14.9419746
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9851913, 17.9847794
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6523285, 18.6519203
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4023666, 15.4000378
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5995331, 19.6013145
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7391548, 18.7392540
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5038681, 20.5033684
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7652206, 24.7664261
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3556747, 20.3557243
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8551712, 20.8557625
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1857491, 16.1844749
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8683167, 21.8673706
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0270195, 19.0220299
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1619568, 18.1621780
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3615494, 31.3609772
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8629913, 19.8634109
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8241844, 21.8241768
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5794144, 23.5821686
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2283478, 36.2280502
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3003845, 29.3035545
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5149994, 32.5099335
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2865677, 26.2863617
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3650284, 23.3663292
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6051750, 16.6088963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4618277, upper bound: 13.5289965
time: 29.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4837351, upper bound: 13.5073515
time: 30.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8094330, 26.8083496
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5951042, 13.5938816
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3091240, 14.3074646
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5856552, 19.5844879
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6369705, 19.6303520
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0914764, 23.0894928
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2134628, 21.2141762
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1441422, 21.1414261
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8965225, 20.8905029
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4172745, 26.4169159
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7325745, 23.7324600
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5803261, 18.5811806
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6989746, 28.6974564
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250916, 30.1244888
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9959106, 37.9916153
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5960350, 18.5919724
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7563934, 36.7521591
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1594238, 21.1596413
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9433365, 14.9430199
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9851761, 17.9846764
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6522217, 18.6520233
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4019661, 15.4004402
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5990601, 19.6017380
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7391090, 18.7392921
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5037308, 20.5034676
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7651291, 24.7665024
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3554611, 20.3559341
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8550949, 20.8557892
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1856270, 16.1845970
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8679810, 21.8675270
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0255241, 19.0235405
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1621323, 18.1619949
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3607941, 31.3617325
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8628235, 19.8635559
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8233910, 21.8249054
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5793610, 23.5822258
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2262421, 36.2300873
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3003616, 29.3035965
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5127182, 32.5123291
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2862778, 26.2866058
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3645477, 23.3667908
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6058121, 16.6081696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4693905, upper bound: 13.5216974
time: 35.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4910544, upper bound: 13.4999171
time: 33.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8083496, 26.8094330
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5938797, 13.5951061
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3074646, 14.3091240
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5844879, 19.5856590
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6303482, 19.6369705
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0894928, 23.0914764
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2141724, 21.2134628
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1414261, 21.1441422
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8905029, 20.8965225
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4169159, 26.4172745
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7324600, 23.7325706
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5811806, 18.5803223
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6974487, 28.6989822
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1244965, 30.1250916
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9916077, 37.9959106
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5919762, 18.5960388
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7521667, 36.7563934
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1596451, 21.1594200
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9430199, 14.9433365
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9846725, 17.9851799
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6520233, 18.6522179
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4004402, 15.4019661
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6017456, 19.5990677
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7392921, 18.7391129
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5034714, 20.5037231
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7665024, 24.7651291
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3559341, 20.3554611
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8557968, 20.8550873
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1845970, 16.1856270
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8675232, 21.8679771
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0235405, 19.0255165
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1619949, 18.1621323
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3617325, 31.3607979
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8635559, 19.8628235
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8249016, 21.8233986
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5822296, 23.5793610
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2300873, 36.2262497
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3035965, 29.3003654
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5123291, 32.5127182
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2866058, 26.2862816
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3667908, 23.3645477
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6081696, 16.6058140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4999171, upper bound: 13.4910544
time: 38.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5216974, upper bound: 13.4693905
time: 37.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8101196, 26.8076630
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5945129, 13.5944729
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3075981, 14.3090248
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5858688, 19.5842438
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6303482, 19.6369476
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0912628, 23.0897217
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2160416, 21.2117691
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1427994, 21.1428833
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8910599, 20.8959656
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4176865, 26.4165039
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7326889, 23.7324028
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5811272, 18.5803909
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6974487, 28.6990128
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1245041, 30.1250801
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9920654, 37.9954529
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5914650, 18.5964432
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7534027, 36.7552109
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1595917, 21.1594696
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9419746, 14.9443817
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9847794, 17.9851913
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6519165, 18.6523285
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4000397, 15.4023685
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.6013184, 19.5995255
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7392540, 18.7391510
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5033646, 20.5038643
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7664261, 24.7652206
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3557205, 20.3556709
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8557663, 20.8551674
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1844749, 16.1857491
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8673706, 21.8683128
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0220299, 19.0270195
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1621704, 18.1619530
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3609695, 31.3615532
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8634109, 19.8629913
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8241844, 21.8241844
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5821686, 23.5794144
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2280426, 36.2283401
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3035507, 29.3003845
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5099258, 32.5149994
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2863617, 26.2865715
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3663254, 23.3650246
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6088943, 16.6051750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5073515, upper bound: 13.4837351
time: 49.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5289964, upper bound: 13.4618277
time: 10.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8082657, 26.8095169
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5938797, 13.5951042
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3066711, 14.3099518
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5857162, 19.5843925
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6302414, 19.6370468
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0914383, 23.0895386
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2173691, 21.2104416
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1427841, 21.1428947
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8902283, 20.8967934
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4175034, 26.4166870
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7329712, 23.7321167
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5812798, 18.5802345
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6996460, 28.6968155
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250992, 30.1244812
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9940186, 37.9934921
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5910301, 18.5968857
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7563019, 36.7522964
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1586685, 21.1603928
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9414482, 14.9449081
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9847946, 17.9851799
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6520920, 18.6521568
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4011307, 15.4012756
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5992737, 19.6015739
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7389412, 18.7394638
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.5033951, 20.5038338
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7647095, 24.7669373
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3551102, 20.3562851
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8557358, 20.8551979
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1855354, 16.1846848
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8694305, 21.8662491
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0220299, 19.0270195
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1621399, 18.1619911
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3609314, 31.3615952
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8633728, 19.8630295
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8234673, 21.8248901
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5805359, 23.5810547
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2257843, 36.2306137
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3005676, 29.3033676
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5086365, 32.5162964
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2861023, 26.2868233
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3647308, 23.3666229
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6079140, 16.6061573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1724

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4993801, upper bound: 13.4917047
time: 32.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5210365, upper bound: 13.4698079
time: 26.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 61.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4698078, upper bound: 13.5210366
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4917047, upper bound: 13.4993801
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4618277, upper bound: 13.5289965
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4837351, upper bound: 13.5073515
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4693905, upper bound: 13.5216974
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4910544, upper bound: 13.4999171
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4999171, upper bound: 13.4910544
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.5216974, upper bound: 13.4693905
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.5073515, upper bound: 13.4837351
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.5289964, upper bound: 13.4618277
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.4993801, upper bound: 13.4917047
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 61.39
Output dim: 12, lower bound: -13.5210365, upper bound: 13.4698079

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8139420, 26.8123932
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6086807, 13.6055870
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3095245, 14.3063087
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5906334, 19.5972977
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6380997, 19.6312561
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0877762, 23.0932236
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1945038, 21.2035332
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1429901, 21.1428757
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.9022713, 20.8953857
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4176025, 26.4184799
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7538223, 23.7585640
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5672874, 18.5657120
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6906128, 28.6942444
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0960388, 30.1011772
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9857483, 37.9856262
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5883484, 18.5819054
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7142334, 36.7243729
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1757812, 21.1702728
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9296036, 14.9230232
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9734955, 17.9707451
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6312180, 18.6269379
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3971214, 15.3961296
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5876503, 19.5824966
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7220535, 18.7181282
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4909363, 20.4878845
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7661591, 24.7595673
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3193283, 20.3114700
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8436432, 20.8418465
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1764526, 16.1756363
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8721886, 21.8746910
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0097504, 19.0012703
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1470184, 18.1492424
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3382721, 31.3414001
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8149605, 19.8232880
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8007164, 21.8028793
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5808754, 23.5803299
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2418289, 36.2359924
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3217697, 29.3162613
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5082092, 32.5018234
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2786102, 26.2788696
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3650970, 23.3634033
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6061974, 16.6087151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4642737, upper bound: 13.4877073
time: 35.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4364770, upper bound: 13.5154703
time: 32.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8120880, 26.8142471
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6080475, 13.6062183
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3085976, 14.3072395
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5904808, 19.5974464
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6380005, 19.6313553
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0879593, 23.0930405
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1958313, 21.2022018
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1429749, 21.1428871
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.9014473, 20.8962097
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4174194, 26.4186630
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7541122, 23.7582779
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5674400, 18.5655556
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6928101, 28.6920471
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0966339, 30.1005783
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9877014, 37.9836731
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5879059, 18.5823441
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7171326, 36.7214584
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1748657, 21.1711922
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9290771, 14.9235497
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9735107, 17.9707298
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6313934, 18.6267624
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3982124, 15.3950348
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5856056, 19.5845490
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7217407, 18.7184410
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4909592, 20.4878578
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7644424, 24.7612762
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3187180, 20.3120842
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8436127, 20.8418770
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1775131, 16.1745758
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8742485, 21.8726273
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0097504, 19.0012665
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1469727, 18.1492805
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3382263, 31.3414383
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8149147, 19.8233337
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8000145, 21.8035812
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5792351, 23.5819702
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2395401, 36.2382584
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3187866, 29.3192482
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5069199, 32.5031128
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2783508, 26.2791214
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3634949, 23.3650017
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6052170, 16.6096954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4563028, upper bound: 13.4956699
time: 39.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4285014, upper bound: 13.5234216
time: 27.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8138580, 26.8124847
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6086807, 13.6055870
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3087006, 14.3071022
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5918999, 19.5960693
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6380234, 19.6313629
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0897141, 23.0912704
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1975250, 21.2003403
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1442337, 21.1415176
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.9019966, 20.8956528
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4181900, 26.4178925
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7542801, 23.7580528
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5673714, 18.5656128
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6927795, 28.6920395
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0966492, 30.1005669
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9881592, 37.9832153
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5875092, 18.5828514
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7183228, 36.7202225
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1748047, 21.1712456
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9280319, 14.9245949
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9734955, 17.9706306
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6312790, 18.6268692
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3978081, 15.3954391
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5851479, 19.5849724
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7217026, 18.7184792
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4908218, 20.4879532
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7643509, 24.7613525
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3185043, 20.3122940
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8435364, 20.8418999
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1773911, 16.1746979
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8739128, 21.8727875
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0082474, 19.0027771
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1471558, 18.1490974
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3374710, 31.3421936
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8147469, 19.8234711
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.7992210, 21.8043098
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5791817, 23.5820236
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2374649, 36.2402954
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3187637, 29.3192902
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5046310, 32.5055161
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2780609, 26.2793655
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3630219, 23.3654633
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6058540, 16.6089706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4638520, upper bound: 13.4883829
time: 28.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4360620, upper bound: 13.5161564
time: 32.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8124847, 26.8138504
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6055870, 13.6086826
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3071022, 14.3086967
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5960655, 19.5918961
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6313629, 19.6380234
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0912704, 23.0897141
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2003403, 21.1975250
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1415176, 21.1442337
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8956490, 20.9020004
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4178925, 26.4181900
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7580490, 23.7542839
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5656090, 18.5673752
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6920471, 28.6927872
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1005707, 30.0966492
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9832153, 37.9881592
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5828552, 18.5875053
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7202148, 36.7183304
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1712494, 21.1748085
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9245987, 14.9280319
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9706268, 17.9734917
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6268692, 18.6312828
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3954391, 15.3978081
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5849724, 19.5851440
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7184830, 18.7216988
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4879532, 20.4908218
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7613525, 24.7643585
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3122940, 20.3185005
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8418961, 20.8435326
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1746979, 16.1773911
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8727837, 21.8739166
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0027771, 19.0082474
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1490936, 18.1471596
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3421936, 31.3374748
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8234749, 19.8147507
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8043098, 21.7992249
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5820274, 23.5791779
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2403030, 36.2374573
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3192902, 29.3187637
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5055161, 32.5046310
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2793655, 26.2780647
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3654633, 23.3630219
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6089706, 16.6058559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5161563, upper bound: 13.4360620
time: 29.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4883829, upper bound: 13.4638521
time: 27.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8142548, 26.8120880
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6062164, 13.6080475
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3072357, 14.3085976
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5974388, 19.5904808
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6313553, 19.6380005
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0930481, 23.0879593
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2022018, 21.1958313
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1428833, 21.1429749
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8962135, 20.9014435
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4186630, 26.4174194
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7582779, 23.7541122
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5655556, 18.5674400
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6920471, 28.6928101
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1005783, 30.0966339
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9836731, 37.9877014
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5823441, 18.5879097
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7214661, 36.7171478
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1711884, 21.1748619
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9235535, 14.9290771
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9707336, 17.9735031
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6267624, 18.6313934
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3950348, 15.3982124
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5845451, 19.5856018
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7184448, 18.7217369
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4878540, 20.4909630
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7612762, 24.7644501
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3120804, 20.3187141
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8418732, 20.8436165
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1745758, 16.1775131
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8726311, 21.8742523
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0012665, 19.0097466
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1492767, 18.1469765
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3414383, 31.3382301
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8233299, 19.8149185
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8035851, 21.8000145
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5819740, 23.5792351
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2382584, 36.2395554
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3192444, 29.3187828
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5031204, 32.5069122
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2791214, 26.2783546
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3649979, 23.3634987
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6096954, 16.6052170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5234216, upper bound: 13.4285014
time: 37.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4956699, upper bound: 13.4563028
time: 26.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8123932, 26.8139420
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6055870, 13.6086807
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3063126, 14.3095245
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5973015, 19.5906296
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6312561, 19.6380997
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0932236, 23.0877762
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2035294, 21.1945038
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1428757, 21.1429901
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8953896, 20.9022713
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4184799, 26.4176025
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7585678, 23.7538261
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5657082, 18.5672874
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6942444, 28.6906128
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1011810, 30.0960388
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9856262, 37.9857483
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5819092, 18.5883522
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7243652, 36.7142334
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1702728, 21.1757812
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9230270, 14.9296036
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9707489, 17.9734917
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6269379, 18.6312218
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3961296, 15.3971195
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5825005, 19.5876541
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7181320, 18.7220497
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4878845, 20.4909325
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7595596, 24.7661591
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3114700, 20.3193245
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8418503, 20.8436432
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1756363, 16.1764526
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8746910, 21.8721886
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0012665, 19.0097466
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1492462, 18.1470146
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3414001, 31.3382721
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8232918, 19.8149605
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8028831, 21.8007202
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5803337, 23.5808716
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2360001, 36.2418289
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3162613, 29.3217697
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5018234, 32.5082092
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2788696, 26.2786064
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3634033, 23.3650970
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6087151, 16.6061993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5154703, upper bound: 13.4364770
time: 28.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4877073, upper bound: 13.4642738
time: 34.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 65.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4642737, upper bound: 13.4877073
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4364770, upper bound: 13.5154703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4563028, upper bound: 13.4956699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4285014, upper bound: 13.5234216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4638520, upper bound: 13.4883829
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4360620, upper bound: 13.5161564
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.5161563, upper bound: 13.4360620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4883829, upper bound: 13.4638521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.5234216, upper bound: 13.4285014
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4956699, upper bound: 13.4563028
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.5154703, upper bound: 13.4364770
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 65.69
Output dim: 12, lower bound: -13.4877073, upper bound: 13.4642738

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8097610, 26.8119202
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6064072, 13.6052990
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3074722, 14.3060760
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5881844, 19.5946960
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6374893, 19.6304588
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0857925, 23.0925064
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1911697, 21.1971664
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1411362, 21.1411667
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.9011345, 20.8957901
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4161453, 26.4222412
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7528305, 23.7592316
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5678368, 18.5653534
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6744080, 28.6789551
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0876694, 30.0968056
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9552765, 37.9623795
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5871735, 18.5809708
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.6963806, 36.7089539
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1738129, 21.1683693
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9287453, 14.9231873
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9734421, 17.9706421
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6280861, 18.6238823
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3970108, 15.3938370
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5899849, 19.5826569
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7243118, 18.7173538
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4910889, 20.4877701
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7647095, 24.7591400
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3152084, 20.3040276
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8461990, 20.8409233
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1696930, 16.1692467
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8739433, 21.8721619
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0090027, 19.0004425
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1399384, 18.1420784
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3357849, 31.3235931
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8119278, 19.8027954
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8055992, 21.7996750
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5833473, 23.5763664
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2395706, 36.2231522
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3254623, 29.3171844
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.5039215, 32.4893570
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2668762, 26.2510071
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3554459, 23.3463821
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5990791, 16.6028862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1704

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4238729, upper bound: 13.5139849
time: 31.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4190805, upper bound: 13.5187896
time: 35.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8119202, 26.8097610
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.6053009, 13.6064053
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3060760, 14.3074722
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5946999, 19.5881882
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6304550, 19.6374855
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0925064, 23.0857925
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1971664, 21.1911736
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1411667, 21.1411400
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8957939, 20.9011345
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4222412, 26.4161453
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7592316, 23.7528267
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5653496, 18.5678368
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6789551, 28.6744080
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0968094, 30.0876694
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9623871, 37.9552689
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5809784, 18.5871658
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7089539, 36.6963806
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1683655, 21.1738129
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9231873, 14.9287453
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9706421, 17.9734459
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6238823, 18.6280899
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3938370, 15.3970108
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5826607, 19.5899811
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7173538, 18.7243156
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4877701, 20.4910851
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7591400, 24.7647095
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3040237, 20.3152084
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8409195, 20.8462029
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1692505, 16.1696930
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8721581, 21.8739471
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0004425, 19.0090027
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1420746, 18.1399460
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3235855, 31.3357849
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8027954, 19.8119278
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.7996788, 21.8055954
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5763664, 23.5833511
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2231522, 36.2395782
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3171844, 29.3254623
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4893570, 32.5039215
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2510071, 26.2668762
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3463821, 23.3554459
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6028862, 16.5990791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1508

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1704

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5187895, upper bound: 13.4190805
time: 16.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5139848, upper bound: 13.4238730
time: 18.57 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 37.06 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 37.06
Output dim: 12, lower bound: -13.4238729, upper bound: 13.5139849
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 37.06
Output dim: 12, lower bound: -13.4190805, upper bound: 13.5187896
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 37.06
Output dim: 12, lower bound: -13.5187895, upper bound: 13.4190805
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 37.06
Output dim: 12, lower bound: -13.5139848, upper bound: 13.4238730

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 49.44 + 1369.92 = 1419.36 seconds
