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
execution time: IAR + RelationalAnalysis = 2.71 + 47.60 = 50.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -13.5325782, upper bound: 13.5325782

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1733

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4848735, upper bound: 13.5288537
time: 27.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5288537, upper bound: 13.4848735
time: 27.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 54.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 54.54
Output dim: 12, lower bound: -13.4848735, upper bound: 13.5288537
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 54.54
Output dim: 12, lower bound: -13.5288537, upper bound: 13.4848735

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8168564, 26.8169937
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5942993, 13.5940418
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3136406, 14.3137283
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6003151, 19.6007614
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6014709, 19.5987320
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1063690, 23.1070938
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2511215, 21.2567101
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1498947, 21.1482506
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8822174, 20.8802071
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4255981, 26.4259338
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7294693, 23.7341003
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5647964, 18.5616379
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6802444, 28.6898193
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1124725, 30.1184311
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0057068, 38.0012436
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6224632, 18.6209641
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7754822, 36.7753525
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1564255, 21.1496162
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9634438, 14.9600067
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9615173, 17.9569588
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6432800, 18.6380081
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4076900, 15.4042206
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5942917, 19.5920601
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7349586, 18.7265892
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4916115, 20.4867897
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7635040, 24.7548828
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3462601, 20.3362732
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8319626, 20.8273315
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1875534, 16.1865158
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8228302, 21.8136864
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0531158, 19.0498085
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1467476, 18.1550407
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3683701, 31.3679543
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8698006, 19.8694763
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8471756, 21.8453674
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5975342, 23.5975266
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2225800, 36.2218018
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3123779, 29.3131752
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4813004, 32.4808578
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2796021, 26.2799149
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3599472, 23.3647575
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6143951, 16.6210995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1404

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4805140, upper bound: 13.5280268
time: 32.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4840463, upper bound: 13.5244948
time: 33.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8170013, 26.8168564
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5940399, 13.5942993
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3137283, 14.3136406
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6007576, 19.6003151
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5987320, 19.6014709
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1070938, 23.1063690
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2567062, 21.2511215
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1482544, 21.1498909
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8802109, 20.8822136
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4259338, 26.4255981
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7341003, 23.7294731
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5616379, 18.5647926
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6898270, 28.6802521
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1184311, 30.1124687
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0012360, 38.0057144
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6209679, 18.6224594
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7753448, 36.7754822
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1496201, 21.1564293
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9600067, 14.9634438
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9569626, 17.9615135
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6380081, 18.6432762
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4042225, 15.4076920
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5920639, 19.5942917
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7265892, 18.7349586
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4867897, 20.4916153
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7548828, 24.7635040
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3362732, 20.3462601
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8273315, 20.8319626
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1865158, 16.1875534
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8136902, 21.8228302
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0498047, 19.0531235
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1550407, 18.1467476
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3679504, 31.3683701
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8694725, 19.8698044
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8453674, 21.8471756
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5975266, 23.5975342
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2218018, 36.2225800
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3131714, 29.3123817
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4808578, 32.4813004
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2799149, 26.2796021
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3647614, 23.3599472
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6211014, 16.6143951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1789

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 948

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5286255, upper bound: 13.4412451
time: 31.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4835990, upper bound: 13.4846448
time: 27.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 61.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 61.14
Output dim: 12, lower bound: -13.4805140, upper bound: 13.5280268
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 61.14
Output dim: 12, lower bound: -13.4840463, upper bound: 13.5244948
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 61.14
Output dim: 12, lower bound: -13.5286255, upper bound: 13.4412451
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 61.14
Output dim: 12, lower bound: -13.4835990, upper bound: 13.4846448

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8169250, 26.8170319
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5944405, 13.5941410
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3123703, 14.3120995
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5986938, 19.5989380
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6007462, 19.5978470
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1038132, 23.1041031
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2506256, 21.2562256
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1484909, 21.1465302
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8810196, 20.8788261
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4256287, 26.4259567
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7294922, 23.7341156
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5647812, 18.5616302
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6790771, 28.6887665
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1114731, 30.1172562
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0041046, 37.9993362
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6223679, 18.6209259
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7756348, 36.7751465
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1564560, 21.1497421
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9634819, 14.9600449
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9615135, 17.9569550
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6426697, 18.6374702
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4070053, 15.4036789
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5941696, 19.5919647
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7342339, 18.7260170
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4914322, 20.4867668
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7634468, 24.7548904
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3462982, 20.3363152
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8313713, 20.8271332
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1863174, 16.1854973
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8229675, 21.8141098
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0532799, 19.0499535
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1462479, 18.1545563
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3662720, 31.3664398
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8656540, 19.8662949
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8449402, 21.8435135
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5966568, 23.5968246
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2199249, 36.2197342
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3109055, 29.3119049
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4813309, 32.4809036
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2780457, 26.2784767
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3579712, 23.3629990
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6134758, 16.6202011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1700

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4440639, upper bound: 13.5269306
time: 30.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4793675, upper bound: 13.4918722
time: 32.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8168945, 26.8170700
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5943985, 13.5941811
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3120117, 14.3124580
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5984955, 19.5991440
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6005859, 19.5980072
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1033783, 23.1045456
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2506409, 21.2562180
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1481705, 21.1468506
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8808289, 20.8790131
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4256210, 26.4259644
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7294922, 23.7341232
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5647888, 18.5616302
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6791992, 28.6886520
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1112976, 30.1174316
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0037994, 37.9996262
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6224136, 18.6208725
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7752838, 36.7755051
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1565552, 21.1496429
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9634819, 14.9600449
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9615135, 17.9569550
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6427460, 18.6374054
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4071503, 15.4035339
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5941925, 19.5919380
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7343864, 18.7258644
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4915848, 20.4866142
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7635155, 24.7548294
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3462982, 20.3363152
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8317604, 20.8267441
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1865387, 16.1852837
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8232574, 21.8138161
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0532722, 19.0499649
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1462631, 18.1545410
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3668518, 31.3658638
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8666229, 19.8653259
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8453217, 21.8431435
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5968323, 23.5966492
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2205048, 36.2191467
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3111115, 29.3116989
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4813385, 32.4808960
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2781677, 26.2783623
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3581924, 23.3627815
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6134949, 16.6201820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 853

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4833436, upper bound: 13.5006242
time: 41.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4601794, upper bound: 13.5237921
time: 33.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8196487, 26.8194351
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5951538, 13.5954857
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3133087, 14.3122368
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6008911, 19.6004257
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5845032, 19.5857735
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1131744, 23.1116562
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2505112, 21.2429886
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1484909, 21.1501503
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8737793, 20.8750877
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4265366, 26.4267349
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7400513, 23.7341957
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5406952, 18.5472717
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6772919, 28.6653824
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1087952, 30.1010895
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0002747, 38.0064621
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6231918, 18.6240273
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7752686, 36.7754211
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1428375, 21.1530609
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9576797, 14.9614105
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9460144, 17.9523849
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6188812, 18.6275940
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4046135, 15.4081764
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5707397, 19.5757751
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7115860, 18.7225418
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4767075, 20.4834557
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7545395, 24.7671432
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3212128, 20.3335991
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8102722, 20.8179855
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1840897, 16.1854553
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7831726, 21.7977104
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0549850, 19.0584755
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1390343, 18.1266174
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3682938, 31.3686867
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8674736, 19.8676910
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8567238, 21.8598633
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5996552, 23.5995560
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2310028, 36.2318573
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3129349, 29.3121262
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4683456, 32.4665833
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2713852, 26.2695770
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3569946, 23.3507004
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6174622, 16.6101494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1334

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5283942, upper bound: 13.4393354
time: 31.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5267216, upper bound: 13.4410137
time: 25.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 58.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.4440639, upper bound: 13.5269306
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.4793675, upper bound: 13.4918722
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.4833436, upper bound: 13.5006242
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.4601794, upper bound: 13.5237921
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.5283942, upper bound: 13.4393354
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 58.80
Output dim: 12, lower bound: -13.5267216, upper bound: 13.4410137

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8133087, 26.8104248
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5930939, 13.5911999
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3123055, 14.3120499
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5960770, 19.5959129
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5975189, 19.5948601
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1024094, 23.1024551
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2437439, 21.2506599
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1444778, 21.1414833
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8762169, 20.8731079
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4223938, 26.4208069
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7293396, 23.7340851
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5577927, 18.5522118
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6686096, 28.6805725
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1046143, 30.1116180
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9990387, 37.9908829
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6202850, 18.6187248
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7755127, 36.7749939
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1564789, 21.1478577
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9633369, 14.9598846
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9579697, 17.9520721
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6346970, 18.6270180
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4056892, 15.4023762
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5866508, 19.5814934
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7291298, 18.7195358
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4878540, 20.4817200
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7612534, 24.7500687
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3388138, 20.3269348
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8273659, 20.8222160
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1857185, 16.1848030
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8141327, 21.8014832
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0518456, 19.0485344
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1334991, 18.1454620
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3648300, 31.3652725
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8653679, 19.8660240
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8449173, 21.8434792
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5945053, 23.5950050
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2203293, 36.2191544
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3106995, 29.3117752
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4747467, 32.4763565
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2766953, 26.2775192
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3529587, 23.3591080
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6121254, 16.6191902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1425

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4438675, upper bound: 13.5254953
time: 26.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4426265, upper bound: 13.5267342
time: 28.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8047714, 26.8020477
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5962906, 13.5947685
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2942467, 14.2914658
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5817719, 19.5797768
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5886955, 19.5811272
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0832520, 23.0807877
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2292175, 21.2388611
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1354446, 21.1316490
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8665543, 20.8617096
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4187622, 26.4177780
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7204361, 23.7240677
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5614510, 18.5599060
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6617508, 28.6745605
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0926666, 30.0959015
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9977417, 37.9924393
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6208954, 18.6189041
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7560120, 36.7528534
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1510010, 21.1446724
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9634628, 14.9600487
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9624710, 17.9584274
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6387367, 18.6346245
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4073887, 15.4039154
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5945511, 19.5942116
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7354660, 18.7276230
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4922104, 20.4886284
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7581100, 24.7498550
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3267860, 20.3189278
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8337250, 20.8306847
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1871033, 16.1871948
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8233337, 21.8166084
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0543671, 19.0512428
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1334114, 18.1440392
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3549500, 31.3571701
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8548508, 19.8560410
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8441925, 21.8423347
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5923691, 23.5930099
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.1947632, 36.1977997
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3075943, 29.3086395
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4816666, 32.4812317
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2515106, 26.2558670
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3387527, 23.3465843
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5898552, 16.6014919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4599009, upper bound: 13.4957069
time: 26.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4319668, upper bound: 13.5235169
time: 29.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8006058, 26.7972794
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5873718, 13.5864964
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3130531, 14.3105469
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6027985, 19.6016502
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5850143, 19.5853767
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1145248, 23.1127090
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2372131, 21.2315521
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1502762, 21.1514893
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8628311, 20.8625336
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4236221, 26.4229050
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7401962, 23.7339516
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5387840, 18.5454407
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6824951, 28.6726532
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1068344, 30.0990372
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0031433, 38.0112839
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6158867, 18.6158218
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7828674, 36.7853622
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1422462, 21.1525154
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9580956, 14.9618034
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9406509, 17.9477272
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6118088, 18.6214638
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3966675, 15.4015808
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5715828, 19.5754166
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7116203, 18.7225761
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4784355, 20.4859734
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7536545, 24.7661514
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3174210, 20.3301277
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8089294, 20.8166962
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1764297, 16.1789932
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7618179, 21.7793045
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0491791, 19.0536003
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1362381, 18.1241570
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3688965, 31.3692245
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8668594, 19.8669167
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8548164, 21.8577232
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5943832, 23.5931168
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2202301, 36.2195053
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2969589, 29.2929420
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4606171, 32.4579315
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2703476, 26.2683640
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3535385, 23.3456001
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6120796, 16.6038208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1381

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1404

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5240351, upper bound: 13.4385072
time: 34.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5275673, upper bound: 13.4349666
time: 31.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7974930, 26.8003845
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5861626, 13.5877056
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3116188, 14.3119850
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6021118, 19.6023407
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5841064, 19.5862846
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1142273, 23.1130066
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2390747, 21.2296867
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1498337, 21.1519356
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8612289, 20.8641357
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4227066, 26.4238205
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7398071, 23.7343407
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5388603, 18.5453568
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6845551, 28.6705933
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1067505, 30.0991211
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0050964, 38.0093384
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6149864, 18.6167297
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7852173, 36.7830124
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1422920, 21.1524773
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9580688, 14.9618263
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9413528, 17.9470253
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6127548, 18.6205139
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3980179, 15.4002266
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5703773, 19.5766220
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7116203, 18.7225761
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4792213, 20.4851837
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7535477, 24.7662582
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3177414, 20.3297997
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8089828, 20.8166351
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1776276, 16.1777954
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7647705, 21.7763596
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0501175, 19.0526695
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1365738, 18.1238213
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3688354, 31.3692856
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8666992, 19.8670731
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8545876, 21.8579521
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5932159, 23.5942802
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2186584, 36.2210770
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2937546, 29.2961502
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4596863, 32.4588623
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2701721, 26.2685356
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3518906, 23.3472443
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6111336, 16.6047649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1304

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1791

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5255755, upper bound: 13.4272498
time: 32.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5129619, upper bound: 13.4398656
time: 36.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 70.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.4438675, upper bound: 13.5254953
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.4426265, upper bound: 13.5267342
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.4599009, upper bound: 13.4957069
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.4319668, upper bound: 13.5235169
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.5240351, upper bound: 13.4385072
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.5275673, upper bound: 13.4349666
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.5255755, upper bound: 13.4272498
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 70.41
Output dim: 12, lower bound: -13.5129619, upper bound: 13.4398656

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8189468, 26.8151855
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5889816, 13.5863800
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3119202, 14.3118095
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5969887, 19.5966225
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5939026, 19.5911598
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1027985, 23.1027527
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2461624, 21.2535439
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1437149, 21.1407967
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8774414, 20.8741417
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4199753, 26.4177475
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7269287, 23.7310829
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5545044, 18.5494499
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6669846, 28.6786804
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1076126, 30.1141090
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0041046, 37.9954300
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6230164, 18.6208839
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7729950, 36.7728653
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1585999, 21.1504211
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9633675, 14.9599648
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9571953, 17.9514198
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6350784, 18.6269379
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4058781, 15.4025288
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5844650, 19.5795135
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7260094, 18.7169724
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4876404, 20.4814682
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7617722, 24.7506638
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3373909, 20.3262215
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8245773, 20.8196030
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1858521, 16.1849174
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8046265, 21.7932091
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0514450, 19.0482063
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1344376, 18.1464500
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3685150, 31.3691406
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8656044, 19.8655930
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8447151, 21.8429451
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5946617, 23.5950699
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2221527, 36.2211990
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3074799, 29.3078232
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4755554, 32.4772186
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2817612, 26.2841110
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3545074, 23.3609085
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6120605, 16.6191807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1541

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4398247, upper bound: 13.5250492
time: 35.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4434177, upper bound: 13.5214508
time: 25.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8180695, 26.8160629
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5882797, 13.5870800
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3120651, 14.3116646
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5967903, 19.5968246
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5938187, 19.5912514
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1027069, 23.1028442
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2466202, 21.2530823
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1437912, 21.1407166
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8772507, 20.8743286
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4193268, 26.4183960
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7263489, 23.7316666
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5550308, 18.5489235
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6667252, 28.6789398
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1071091, 30.1146126
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0035858, 37.9959412
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6224518, 18.6214485
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7733765, 36.7724915
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1590424, 21.1499786
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9634171, 14.9599152
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9573174, 17.9512978
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6346130, 18.6274071
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4058399, 15.4025650
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5846710, 19.5793114
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7265663, 18.7164154
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4876022, 20.4815063
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7618561, 24.7505798
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3381004, 20.3255119
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8247604, 20.8194199
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1858292, 16.1849365
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8058624, 21.7919731
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0515060, 19.0481453
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1344833, 18.1464043
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3686981, 31.3689575
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8649330, 19.8662643
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8443794, 21.8432732
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5945702, 23.5951614
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2223969, 36.2209549
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3067398, 29.3085632
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4756088, 32.4771576
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2832870, 26.2825890
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3547668, 23.3606491
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6121178, 16.6191235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 521

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4323909, upper bound: 13.5263191
time: 31.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4422124, upper bound: 13.5165011
time: 33.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8093948, 26.8072891
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5958366, 13.5943298
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2873268, 14.2858772
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5805664, 19.5787773
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5674782, 19.5624771
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0895233, 23.0885315
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2329102, 21.2447281
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1375275, 21.1339989
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8542366, 20.8506241
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4213638, 26.4198914
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7222137, 23.7265472
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5436249, 18.5385475
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6499329, 28.6648788
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0803146, 30.0855980
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0012970, 37.9957809
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6176376, 18.6161270
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7485199, 36.7456284
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1419640, 21.1323662
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9623718, 14.9587021
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9560432, 17.9506721
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6227493, 18.6154671
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4086304, 15.4047928
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5729828, 19.5683327
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7226715, 18.7120667
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4851074, 20.4796257
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7447968, 24.7324982
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3174210, 20.3076096
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8179779, 20.8115807
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1864090, 16.1863251
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7980652, 21.7859535
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0617905, 19.0583115
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1300392, 18.1444664
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3609009, 31.3630981
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8281136, 19.8292122
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8596992, 21.8555756
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5973320, 23.5979233
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2112045, 36.2126617
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3099670, 29.3110657
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4739304, 32.4749146
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2481995, 26.2530899
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3378372, 23.3458328
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5897408, 16.6015034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1283

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4315905, upper bound: 13.5224514
time: 40.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4318967, upper bound: 13.5234517
time: 43.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8006744, 26.7973175
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5875015, 13.5865841
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3117867, 14.3089180
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6011925, 19.5998268
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5842972, 19.5844955
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1119766, 23.1097260
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2367096, 21.2310600
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1488800, 21.1497650
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8616371, 20.8611526
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4236526, 26.4229202
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7402267, 23.7339668
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5387764, 18.5454369
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6813278, 28.6715927
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1058273, 30.0978546
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0015564, 38.0093842
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6157990, 18.6157761
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7830353, 36.7851562
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1422882, 21.1526527
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9581337, 14.9618416
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9406586, 17.9477310
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6111984, 18.6209259
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3959770, 15.4010353
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5714607, 19.5753098
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7108994, 18.7220078
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4782410, 20.4859390
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7536049, 24.7661591
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3174706, 20.3301773
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8083420, 20.8165054
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1752014, 16.1779785
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7619324, 21.7797165
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0493355, 19.0537453
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1357346, 18.1236725
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3668060, 31.3677063
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8627090, 19.8637390
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8525925, 21.8558731
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5935097, 23.5924263
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2175980, 36.2174683
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2954865, 29.2916718
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4606476, 32.4579773
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2687912, 26.2669182
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3515472, 23.3438263
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6111679, 16.6029282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1789

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5224632, upper bound: 13.4330251
time: 30.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5185566, upper bound: 13.4369337
time: 14.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8006363, 26.7973557
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5874634, 13.5866222
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3114243, 14.3092766
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6009789, 19.6000366
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5841370, 19.5846558
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1115341, 23.1101608
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2367096, 21.2310562
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1485519, 21.1500893
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8614388, 20.8613396
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4236450, 26.4229355
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7402115, 23.7339783
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5387764, 18.5454330
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6814499, 28.6714859
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1056519, 30.0980301
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0012512, 38.0096817
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6158447, 18.6157265
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7826691, 36.7855225
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1423950, 21.1525497
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9581337, 14.9618416
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9406586, 17.9477310
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6112671, 18.6208611
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3961182, 15.4008923
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5714912, 19.5752831
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7110519, 18.7218590
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4783936, 20.4857864
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7536659, 24.7660980
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3174706, 20.3301811
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8087387, 20.8161163
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1754227, 16.1777649
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7622375, 21.7794228
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0493279, 19.0537529
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1357498, 18.1236572
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3673782, 31.3671341
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8636780, 19.8627701
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8529663, 21.8554993
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5936928, 23.5922508
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2181778, 36.2168732
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2956924, 29.2914658
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4606628, 32.4579697
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2689056, 26.2668037
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3517609, 23.3436089
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6111870, 16.6029091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1428

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1334

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5190753, upper bound: 13.4340380
time: 37.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5266305, upper bound: 13.4264811
time: 35.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7908859, 26.7949219
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5829926, 13.5850563
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3077202, 14.3087463
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5982475, 19.5991592
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5769577, 19.5803032
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1092300, 23.1089020
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2312317, 21.2204361
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1431732, 21.1463470
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8534126, 20.8577232
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4196625, 26.4212646
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7362518, 23.7314529
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5388680, 18.5453606
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6791763, 28.6641846
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1041794, 30.0970039
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9940338, 38.0000534
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6125412, 18.6146164
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7746277, 36.7741852
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1418686, 21.1519814
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9579086, 14.9616203
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9414520, 17.9470062
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6126785, 18.6204338
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3980179, 15.4001961
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5703316, 19.5763168
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7113762, 18.7222710
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4790382, 20.4847603
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7530746, 24.7657471
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3175583, 20.3294716
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8085861, 20.8157158
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1774330, 16.1775818
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7628555, 21.7745056
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0498352, 19.0522575
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1309967, 18.1171837
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3656921, 31.3655472
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8640022, 19.8638535
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8513908, 21.8541412
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5876999, 23.5877037
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2158356, 36.2177200
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2873993, 29.2885818
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4585266, 32.4576492
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2671127, 26.2648926
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3451614, 23.3392181
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6043167, 16.5966911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 685

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5248606, upper bound: 13.3932168
time: 27.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4915493, upper bound: 13.4265275
time: 37.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 67.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4398247, upper bound: 13.5250492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4434177, upper bound: 13.5214508
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4323909, upper bound: 13.5263191
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4422124, upper bound: 13.5165011
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4315905, upper bound: 13.5224514
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4318967, upper bound: 13.5234517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.5224632, upper bound: 13.4330251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.5185566, upper bound: 13.4369337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.5190753, upper bound: 13.4340380
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.5266305, upper bound: 13.4264811
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.5248606, upper bound: 13.3932168
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.21
Output dim: 12, lower bound: -13.4915493, upper bound: 13.4265275

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8080978, 26.8020325
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5856895, 13.5819969
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3078880, 14.3070374
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5879822, 19.5867157
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5985336, 19.5944633
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0921249, 23.0903015
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2381363, 21.2466698
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1345901, 21.1298943
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8596725, 20.8531799
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4183044, 26.4148788
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7237854, 23.7260132
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5535240, 18.5485649
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6661911, 28.6779022
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1080933, 30.1142159
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9815063, 37.9682922
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6119995, 18.6081352
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7616730, 36.7590027
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1523590, 21.1450272
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9617958, 14.9589577
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9599838, 17.9550819
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6336327, 18.6257477
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4058037, 15.4024677
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5865288, 19.5819893
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7219849, 18.7136536
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4895401, 20.4839363
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7562523, 24.7457733
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3335953, 20.3225479
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8298302, 20.8267059
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1852341, 16.1841850
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8059540, 21.7946663
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0520248, 19.0490074
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1303711, 18.1430244
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3645172, 31.3657951
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8580894, 19.8593063
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8343735, 21.8342896
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5795822, 23.5824051
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2217331, 36.2208328
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2954865, 29.2977333
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4756088, 32.4772949
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2817764, 26.2841263
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3452606, 23.3531647
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6110916, 16.6179123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1713

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4384525, upper bound: 13.5194867
time: 35.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4342999, upper bound: 13.5236698
time: 30.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8057938, 26.8043365
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5845985, 13.5830936
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3071518, 14.3077774
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5870819, 19.5876083
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5972061, 19.5957909
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0903473, 23.0920792
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2392960, 21.2455063
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1328125, 21.1316757
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8564758, 20.8563766
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4171066, 26.4160690
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7218628, 23.7279472
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5536156, 18.5484657
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6662064, 28.6779022
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1077194, 30.1145859
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9769592, 37.9728317
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6102676, 18.6098633
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7591400, 36.7615433
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1532135, 21.1441841
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9623604, 14.9583931
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9608536, 17.9542084
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6338921, 18.6254959
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4058189, 15.4024506
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5869408, 19.5815773
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7226868, 18.7129517
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4901199, 20.4833603
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7568779, 24.7451477
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3337173, 20.3224258
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8316765, 20.8248596
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1851120, 16.1843033
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8060837, 21.7945404
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0522537, 19.0487785
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1310120, 18.1423836
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3651733, 31.3651428
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8593178, 19.8580780
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8360596, 21.8326073
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5820007, 23.5799904
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2217941, 36.2207870
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2973938, 29.2958298
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4756393, 32.4772644
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2817764, 26.2841263
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3467636, 23.3516655
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6107903, 16.6182117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1393

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 589

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4427172, upper bound: 13.5208306
time: 25.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4427974, upper bound: 13.5207511
time: 28.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8127975, 26.8107224
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5842781, 13.5818539
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3066101, 14.3049622
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5980072, 19.5982590
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6032829, 19.5970764
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1003876, 23.1001968
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2387085, 21.2503929
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1342087, 21.1292534
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8639755, 20.8586578
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4218674, 26.4205093
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7241211, 23.7297974
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5510330, 18.5450096
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6473465, 28.6628494
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1066132, 30.1139832
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9836731, 37.9719315
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6227798, 18.6201439
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7633209, 36.7602692
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1540375, 21.1431351
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9649620, 14.9596939
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9551086, 17.9483681
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6323700, 18.6230049
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4099007, 15.4049168
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5845222, 19.5791931
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7230301, 18.7106590
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4831696, 20.4761734
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7683105, 24.7546997
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3285217, 20.3150826
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8242531, 20.8186378
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1843605, 16.1827240
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7993126, 21.7851868
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0452118, 19.0402222
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1084671, 18.1246338
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3529205, 31.3557739
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8419342, 19.8468208
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8378487, 21.8376656
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5880928, 23.5903893
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2118301, 36.2121277
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3101273, 29.3140640
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4731979, 32.4747162
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2612762, 26.2641029
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3293686, 23.3394165
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5873337, 16.6001053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1476

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4299826, upper bound: 13.5263153
time: 29.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4323507, upper bound: 13.5228449
time: 34.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8086243, 26.8064270
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5958252, 13.5943184
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2806740, 14.2780685
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5796776, 19.5775146
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5618668, 19.5553589
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0868607, 23.0844193
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2449112, 21.2554359
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1342468, 21.1299095
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8481827, 20.8437157
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4228668, 26.4214630
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7221451, 23.7261314
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5331154, 18.5299835
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6504059, 28.6651154
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0721054, 30.0755577
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0067444, 38.0016022
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6185379, 18.6169472
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7410278, 36.7370605
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1314087, 21.1240234
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9581642, 14.9552727
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9506950, 17.9463577
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6064110, 18.6022186
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4046822, 15.4014397
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5686951, 19.5665970
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7111435, 18.7025719
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4816742, 20.4781151
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7300110, 24.7207260
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3086014, 20.3003616
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8126183, 20.8085899
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1845093, 16.1850319
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7806854, 21.7720566
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0624466, 19.0590172
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1471024, 18.1589394
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3570480, 31.3602600
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8104324, 19.8132286
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8546600, 21.8535385
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5971184, 23.5982437
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2159729, 36.2197647
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3092270, 29.3108597
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4740067, 32.4748840
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2498093, 26.2541008
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3394012, 23.3470840
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5931625, 16.6044502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1304

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4311273, upper bound: 13.5202843
time: 32.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4294196, upper bound: 13.5219905
time: 33.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8085327, 26.8065262
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5958252, 13.5943203
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2795219, 14.2791824
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5793037, 19.5778923
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5603638, 19.5568657
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0854034, 23.0858307
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2436218, 21.2568436
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1334305, 21.1307220
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8473282, 20.8445702
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4229965, 26.4213943
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7217941, 23.7264786
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5350685, 18.5280304
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6501617, 28.6653519
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0702744, 30.0773888
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0071106, 38.0012283
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6184540, 18.6170311
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7399445, 36.7381363
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1336212, 21.1218109
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9589386, 14.9544945
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9517250, 17.9453278
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6095009, 18.5991287
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4052773, 15.4008408
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5712433, 19.5640488
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7131729, 18.7005386
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4835968, 20.4761925
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7330246, 24.7177124
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3101654, 20.2987938
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8150139, 20.8062286
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1851120, 16.1844292
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7841644, 21.7685776
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0625000, 19.0589676
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1445084, 18.1615257
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3582764, 31.3592529
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8125153, 19.8115311
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8576660, 21.8505440
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5977592, 23.5977058
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2183228, 36.2174072
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3097610, 29.3103256
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4738998, 32.4749908
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2492142, 26.2546921
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3390884, 23.3473969
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5926857, 16.6049500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1428

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4110552, upper bound: 13.5031291
time: 31.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4110552, upper bound: 13.5031291
time: 29.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7912445, 26.7920685
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5831490, 13.5841103
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3084984, 14.3067741
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5963173, 19.5971527
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5812454, 19.5822220
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1038208, 23.1052246
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2341156, 21.2277107
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1409912, 21.1454201
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8556557, 20.8573723
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4182663, 26.4199524
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7363663, 23.7313423
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5384178, 18.5451698
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6803055, 28.6697845
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1008911, 30.0946198
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9880524, 38.0019608
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6157913, 18.6157036
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7702026, 36.7780991
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1410789, 21.1509666
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9562569, 14.9584312
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9405975, 17.9476662
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6110916, 18.6207695
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3950882, 15.3995285
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5695496, 19.5726700
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7091980, 18.7192535
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4772224, 20.4841614
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7520103, 24.7638550
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3149834, 20.3266449
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8055038, 20.8123093
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1751747, 16.1779404
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7598343, 21.7778778
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0467758, 19.0490875
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1326332, 18.1180382
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3611526, 31.3574486
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8571930, 19.8537674
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8472137, 21.8461075
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5888519, 23.5839729
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2125549, 36.2083359
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2910233, 29.2835541
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4552307, 32.4481964
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2647324, 26.2595520
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3463669, 23.3344193
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6086884, 16.5992069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 685

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 785

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5218132, upper bound: 13.4329498
time: 33.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5223874, upper bound: 13.4323710
time: 30.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7927322, 26.7877350
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5827713, 13.5811501
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3079758, 14.3053246
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6004868, 19.5994530
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5844574, 19.5846100
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1109467, 23.1092148
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2327347, 21.2275658
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1461487, 21.1467819
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8550491, 20.8539352
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4220810, 26.4210281
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7403946, 23.7344742
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5377159, 18.5439987
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6741486, 28.6654816
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1054306, 30.0979805
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0004730, 38.0088501
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6149788, 18.6146431
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7827148, 36.7855759
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1429863, 21.1528816
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9582253, 14.9618950
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9404449, 17.9473000
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6088104, 18.6185341
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3932686, 15.3982506
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5699883, 19.5733376
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7088852, 18.7193413
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4785843, 20.4859886
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7532883, 24.7653427
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3175354, 20.3295212
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8081894, 20.8155022
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1758194, 16.1783257
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7613678, 21.7786484
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0493279, 19.0537186
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1290970, 18.1176720
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3618927, 31.3621559
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8579216, 19.8576050
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8477936, 21.8512115
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5930557, 23.5915565
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2178497, 36.2165146
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2963104, 29.2916794
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4585724, 32.4554825
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2683563, 26.2662277
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3507233, 23.3426056
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6111870, 16.6029091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5158677, upper bound: 13.4330451
time: 32.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5180796, upper bound: 13.4308279
time: 29.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7910156, 26.7894516
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5819893, 13.5819340
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3074722, 14.3058281
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6003952, 19.5995369
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5840912, 19.5849762
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1105881, 23.1095657
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2332230, 21.2270775
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1452484, 21.1476784
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8540421, 20.8549461
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4217377, 26.4213715
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7407150, 23.7341499
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5373421, 18.5443764
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6754303, 28.6641922
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1055984, 30.0978127
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0004272, 38.0088882
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6147652, 18.6148567
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7827301, 36.7855530
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1427116, 21.1531410
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9581833, 14.9619331
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9402237, 17.9475136
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6089401, 18.6184044
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3934784, 15.3980408
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5695457, 19.5737877
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7085342, 18.7196922
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4785995, 20.4859772
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7529068, 24.7657242
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3168106, 20.3302460
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8081207, 20.8155670
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1759796, 16.1781616
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7614517, 21.7785606
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0492897, 19.0537567
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1297607, 18.1170082
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3623962, 31.3616486
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8585167, 19.8570137
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8486786, 21.8503342
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5929947, 23.5916100
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2178192, 36.2165451
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2959061, 29.2920837
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4581833, 32.4558792
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2683334, 26.2662544
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3507538, 23.3425751
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6111870, 16.6029091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5240069, upper bound: 13.4218636
time: 30.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5220395, upper bound: 13.4238242
time: 39.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7847137, 26.7888336
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5821609, 13.5854874
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3066788, 14.3079834
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5945282, 19.5913048
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5754471, 19.5809860
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1072235, 23.1059036
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2311859, 21.2200050
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1411285, 21.1452332
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8525200, 20.8573837
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4152756, 26.4162750
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7383194, 23.7240753
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5356712, 18.5431137
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6678925, 28.6465073
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0948868, 30.0824356
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9874573, 37.9953461
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6116524, 18.6144753
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7649689, 36.7590714
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1267319, 21.1429291
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9469414, 14.9548492
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9227905, 17.9349289
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6038055, 18.6147690
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3925591, 15.3970585
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5701637, 19.5762138
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6961479, 18.7125511
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4672432, 20.4772301
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7364273, 24.7557755
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.2883759, 20.3108482
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.7964249, 20.8076477
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1753693, 16.1761551
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7608490, 21.7729263
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0339584, 19.0422592
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1245117, 18.1070251
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3543472, 31.3488159
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8454666, 19.8348961
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8393173, 21.8398323
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5866623, 23.5894356
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2009430, 36.2021179
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2824554, 29.2880859
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4551544, 32.4538574
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2608948, 26.2551422
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3401108, 23.3312950
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6036530, 16.5914650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1427

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5098358, upper bound: 13.3849103
time: 25.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5167718, upper bound: 13.3780511
time: 37.87 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 65.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4384525, upper bound: 13.5194867
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4342999, upper bound: 13.5236698
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4427172, upper bound: 13.5208306
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4427974, upper bound: 13.5207511
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4299826, upper bound: 13.5263153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4323507, upper bound: 13.5228449
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4311273, upper bound: 13.5202843
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4294196, upper bound: 13.5219905
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4110552, upper bound: 13.5031291
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.4110552, upper bound: 13.5031291
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5218132, upper bound: 13.4329498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5223874, upper bound: 13.4323710
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5158677, upper bound: 13.4330451
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5180796, upper bound: 13.4308279
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5240069, upper bound: 13.4218636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5220395, upper bound: 13.4238242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5098358, upper bound: 13.3849103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 65.12
Output dim: 12, lower bound: -13.5167718, upper bound: 13.3780511

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7779770, 26.7813339
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5724564, 13.5724792
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2931442, 14.2964592
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5859604, 19.5850906
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5879669, 19.5886345
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0823669, 23.0827255
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2209778, 21.2251396
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1196442, 21.1187820
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8397903, 20.8381081
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4112167, 26.4097443
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7151260, 23.7207680
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5526581, 18.5473785
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6608200, 28.6726379
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0959167, 30.1060028
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9712830, 37.9601593
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6032486, 18.6016731
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7498779, 36.7500381
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1454163, 21.1348305
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9594879, 14.9556847
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9513550, 17.9422264
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6217041, 18.6087646
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4015732, 15.3955116
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5866776, 19.5810356
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7170563, 18.7068825
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4925537, 20.4821320
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7469254, 24.7322617
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3226814, 20.3093338
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8302727, 20.8223000
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1841812, 16.1826897
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7949448, 21.7764435
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0481796, 19.0430222
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1269531, 18.1399956
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3489456, 31.3431091
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8377914, 19.8301849
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8149567, 21.8061981
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5689392, 23.5664825
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2207184, 36.2185974
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2902603, 29.2899246
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4712296, 32.4715195
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2772369, 26.2795448
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3426743, 23.3501320
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6062889, 16.6133308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1728

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1651

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4280491, upper bound: 13.5100131
time: 32.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4289960, upper bound: 13.5090742
time: 27.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7874069, 26.7719040
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5761719, 13.5687637
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2973061, 14.2922974
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5863571, 19.5846977
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5926971, 19.5839005
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0845566, 23.0805435
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2165985, 21.2295113
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1234741, 21.1149483
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8445969, 20.8333015
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4131699, 26.4077911
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7185364, 23.7173500
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5523376, 18.5476952
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6609421, 28.6725235
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0998764, 30.1020432
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9733734, 37.9580688
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6055374, 18.5993805
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7527161, 36.7471924
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1421661, 21.1380730
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9585228, 14.9566498
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9471207, 17.9464607
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6166534, 18.6138115
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3988495, 15.3982391
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5855789, 19.5821342
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7152176, 18.7087173
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4877319, 20.4869499
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7427444, 24.7364426
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3203773, 20.3116417
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8254280, 20.8271484
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1837387, 16.1831322
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7877274, 21.7836609
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0460434, 19.0451698
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1273422, 18.1396027
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3418350, 31.3502197
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8289719, 19.8390045
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8062820, 21.8148689
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5636597, 23.5717659
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2194977, 36.2198105
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2876740, 29.2925034
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4698257, 32.4729080
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2771988, 26.2795868
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3422318, 23.3505821
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6065102, 16.6131115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1620

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1548

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4330241, upper bound: 13.5224547
time: 30.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4330241, upper bound: 13.5224547
time: 10.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8042679, 26.8032684
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5842094, 13.5827904
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3041878, 14.3057251
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5716858, 19.5775909
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5931778, 19.5925369
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0755539, 23.0822372
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2248001, 21.2345581
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1274414, 21.1281128
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8461533, 20.8496170
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4146194, 26.4144211
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7214890, 23.7286072
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5465775, 18.5383530
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6660843, 28.6766663
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1043701, 30.1122360
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9718018, 37.9685364
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6098709, 18.6096802
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7510986, 36.7557983
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1492615, 21.1383972
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9577103, 14.9516144
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9606705, 17.9538765
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6334496, 18.6249657
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4058075, 15.4023514
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5797501, 19.5725517
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7181931, 18.7063446
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4900742, 20.4830933
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7545815, 24.7422180
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3319855, 20.3198738
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8277855, 20.8190842
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1822014, 16.1800270
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8059387, 21.7941017
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0502739, 19.0458755
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1284714, 18.1402016
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3575058, 31.3579216
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8559113, 19.8567810
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8310623, 21.8272781
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5789948, 23.5769005
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2226562, 36.2198486
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2930374, 29.2928009
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4703979, 32.4722214
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2780304, 26.2807159
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3464203, 23.3511505
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6040459, 16.6136837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1462

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 853

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4420162, upper bound: 13.4969684
time: 34.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4188459, upper bound: 13.5201286
time: 39.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8047333, 26.8028030
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5842896, 13.5827103
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3050995, 14.3048172
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5770645, 19.5722160
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5939484, 19.5917625
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0805054, 23.0772934
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2283478, 21.2310104
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1292496, 21.1262970
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8497162, 20.8460541
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4154587, 26.4135818
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7225189, 23.7275772
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5435028, 18.5414238
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6649704, 28.6777802
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1053619, 30.1112366
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9726868, 37.9676819
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6100845, 18.6094627
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7534027, 36.7535019
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1474152, 21.1402435
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9555817, 14.9537430
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9605179, 17.9540291
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6333656, 18.6250496
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4057198, 15.4024391
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5779190, 19.5743866
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7160797, 18.7084541
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4898453, 20.4833221
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7539558, 24.7428513
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3311615, 20.3206902
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8259010, 20.8209724
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1808357, 16.1813927
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8056488, 21.7943916
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0493431, 19.0468025
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1288376, 18.1398430
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3579483, 31.3574753
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8580246, 19.8546753
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8307266, 21.8276100
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5789032, 23.5769882
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2208405, 36.2216492
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2943649, 29.2914696
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4705963, 32.4720230
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2783661, 26.2803802
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3462448, 23.3513184
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6062660, 16.6114655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 804

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1550

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4420172, upper bound: 13.5206354
time: 23.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4426829, upper bound: 13.5199624
time: 32.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8037491, 26.8032913
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5823898, 13.5804119
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3003502, 14.2996826
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5964813, 19.5983124
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5983658, 19.5929832
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1027145, 23.1040039
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2380600, 21.2498474
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1339188, 21.1295052
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8576126, 20.8537979
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4192047, 26.4183807
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7203903, 23.7276115
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5466309, 18.5393028
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6475067, 28.6630325
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0988541, 30.1077347
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9865265, 37.9742889
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6208725, 18.6186447
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7621918, 36.7593155
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1478958, 21.1354485
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9621925, 14.9561501
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9498177, 17.9418297
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6258965, 18.6149864
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4060287, 15.3999672
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5824585, 19.5764389
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7172203, 18.7033081
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4758682, 20.4674454
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7610168, 24.7458344
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3200493, 20.3043900
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8181763, 20.8109856
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1802902, 16.1776924
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7889099, 21.7720604
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0414658, 19.0357552
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1044159, 18.1217079
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3528290, 31.3556747
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8430176, 19.8478165
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8384018, 21.8380814
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5881081, 23.5904007
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2114258, 36.2116928
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3091049, 29.3134003
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4683304, 32.4707718
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2592773, 26.2623711
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3256683, 23.3365936
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5825272, 16.5963497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1317

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4276382, upper bound: 13.5239756
time: 24.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4276382, upper bound: 13.5239756
time: 25.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8053665, 26.8016739
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5828400, 13.5799618
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3013306, 14.2987022
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5980530, 19.5967331
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5988846, 19.5921593
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1042023, 23.1025238
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2381516, 21.2497559
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1343842, 21.1289597
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8591080, 20.8522987
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4196854, 26.4178391
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7218094, 23.7260628
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5453262, 18.5406075
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6475372, 28.6630020
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1003418, 30.1062202
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9860382, 37.9747009
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6212692, 18.6182365
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7623596, 36.7591400
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1463470, 21.1369934
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9614182, 14.9569244
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9485741, 17.9429665
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6243477, 18.6165314
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4049492, 15.4010448
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5817719, 19.5770950
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7156792, 18.7047462
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4744415, 20.4688263
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7594452, 24.7472839
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3178291, 20.3064461
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8166046, 20.8125381
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1793289, 16.1786575
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7861938, 21.7747879
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0407486, 19.0364799
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1055374, 18.1205750
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3528214, 31.3556786
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8429260, 19.8479004
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8382645, 21.8382187
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5881081, 23.5904083
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2113953, 36.2117233
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3094635, 29.3130417
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4691925, 32.4698563
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2595444, 26.2621078
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3265381, 23.3357201
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5835762, 16.5952988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 866

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 978

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4312635, upper bound: 13.5227624
time: 28.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4322665, upper bound: 13.5217050
time: 29.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8084946, 26.8063049
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5955315, 13.5941086
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2806015, 14.2781601
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5790863, 19.5768967
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5563507, 19.5504494
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0868530, 23.0844116
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2380829, 21.2476807
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1341858, 21.1298523
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8475037, 20.8431396
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4222565, 26.4209824
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7214813, 23.7255363
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5328140, 18.5295944
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6476974, 28.6620255
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0704346, 30.0740967
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0067444, 38.0017395
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6168594, 18.6154785
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7410126, 36.7371979
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1315155, 21.1243057
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9581490, 14.9555817
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9505539, 17.9462318
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6038628, 18.5999680
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4017525, 15.3988495
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5681839, 19.5660172
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7104912, 18.7021637
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4807053, 20.4772377
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7299995, 24.7210541
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3077087, 20.2992477
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8127975, 20.8087997
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1844978, 16.1850433
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7806740, 21.7720451
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0609741, 19.0577621
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1428795, 18.1541176
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3562698, 31.3593674
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8103523, 19.8131332
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8550491, 21.8544006
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5970383, 23.5981064
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2139587, 36.2174759
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3090515, 29.3106384
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4741516, 32.4751587
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2439346, 26.2474442
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3357239, 23.3428688
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5845032, 16.5947037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4169280, upper bound: 13.5157832
time: 33.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4267220, upper bound: 13.5060751
time: 32.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8084946, 26.8062973
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5956154, 13.5940266
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2807655, 14.2779961
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5790634, 19.5769196
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5569611, 19.5498390
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0868530, 23.0844116
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2371521, 21.2486038
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1341934, 21.1298485
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8476105, 20.8430328
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4223938, 26.4208450
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7215576, 23.7254601
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5327225, 18.5296860
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6473312, 28.6623917
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0706482, 30.0738869
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0068817, 38.0015945
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6170654, 18.6152687
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7411652, 36.7370605
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1316910, 21.1241302
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9584732, 14.9552536
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9505692, 17.9462204
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6041679, 18.5996628
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4020882, 15.3985100
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5681076, 19.5660858
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7107277, 18.7019234
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4807968, 20.4771461
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7303352, 24.7207184
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3074799, 20.2994614
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8128357, 20.8087654
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1845207, 16.1850204
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7806740, 21.7720490
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0611877, 19.0575523
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1422844, 18.1547127
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3561554, 31.3594818
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8103371, 19.8131485
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8555298, 21.8539200
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5969849, 23.5981674
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2136688, 36.2177658
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3090134, 29.3106766
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4742737, 32.4750290
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2431488, 26.2482338
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3351822, 23.3434105
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5834160, 16.5957890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4223608, upper bound: 13.5141103
time: 27.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4214971, upper bound: 13.5149854
time: 34.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7883835, 26.7918015
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5772705, 13.5796146
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3038826, 14.3031883
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6002731, 19.6015930
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5761490, 19.5788536
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1054001, 23.1070862
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2227249, 21.2151756
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1440430, 21.1488457
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8425980, 20.8466721
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4213562, 26.4233932
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7363892, 23.7313309
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5380135, 18.5445175
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6979980, 28.6847687
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1013870, 30.0948868
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9890747, 38.0026169
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6127014, 18.6142807
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7830353, 36.7892761
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1388397, 21.1490211
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9548721, 14.9575424
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9408798, 17.9479294
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6121902, 18.6217499
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3961830, 15.4004593
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5639191, 19.5677681
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7043762, 18.7143211
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4778748, 20.4847565
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7440834, 24.7572479
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3151588, 20.3271446
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8057175, 20.8124695
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1792107, 16.1815033
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7502136, 21.7671089
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0463486, 19.0486755
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1291962, 18.1149178
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3600845, 31.3563538
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8545303, 19.8509598
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8414612, 21.8404045
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5839539, 23.5799942
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2047882, 36.2018509
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2737732, 29.2688904
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4459991, 32.4403458
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2651978, 26.2601013
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3429260, 23.3313408
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6088753, 16.5994396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5215378, upper bound: 13.4162350
time: 25.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4933930, upper bound: 13.4324770
time: 28.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7909775, 26.7892075
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5786514, 13.5782299
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3049126, 14.3021584
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6007614, 19.6011047
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5778732, 19.5771255
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1056747, 23.1068039
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2215729, 21.2163239
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1444168, 21.1484680
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8449631, 20.8443108
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4217072, 26.4230499
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7363586, 23.7313614
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5377693, 18.5447617
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6952972, 28.6874847
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1011581, 30.0951157
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9887085, 38.0029678
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6143646, 18.6126175
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7813873, 36.7909241
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1391449, 21.1487312
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9553680, 14.9570465
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9408646, 17.9479446
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6120682, 18.6218719
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3960190, 15.4006271
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5646362, 19.5670395
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7042694, 18.7144318
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4778137, 20.4848137
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7454033, 24.7559280
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3154793, 20.3268127
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8056717, 20.8125191
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1787376, 16.1819801
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7490692, 21.7682610
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0463638, 19.0486603
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1295166, 18.1145935
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3600616, 31.3563766
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8543854, 19.8511047
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8415146, 21.8403549
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5848694, 23.5790749
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2060699, 36.2005692
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2763596, 29.2663040
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4473801, 32.4389648
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2652817, 26.2600174
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3432846, 23.3309822
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6089211, 16.5993938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 965

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4967481, upper bound: 13.4319391
time: 40.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5219300, upper bound: 13.4067603
time: 26.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7850113, 26.7863922
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5795517, 13.5806255
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3043556, 14.3041382
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5932693, 19.5959167
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5806274, 19.5816689
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1046143, 23.1067047
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2327805, 21.2267914
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1414948, 21.1456375
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8461227, 20.8508911
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4190063, 26.4199600
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7393723, 23.7340546
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5351028, 18.5416756
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6741867, 28.6609650
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1043167, 30.0971146
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9968567, 38.0064163
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6129723, 18.6138725
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7809753, 36.7843323
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1413803, 21.1512833
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9576035, 14.9602242
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9401627, 17.9473953
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6087112, 18.6180573
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3927383, 15.3965950
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5677719, 19.5723839
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7073097, 18.7177162
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4781418, 20.4853058
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7525101, 24.7652588
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3163528, 20.3294525
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8068924, 20.8132362
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1747856, 16.1758194
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7603722, 21.7772598
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0486984, 19.0521660
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1294594, 18.1165466
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3615570, 31.3608704
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8578720, 19.8573151
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8486176, 21.8502808
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5929756, 23.5915909
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2172546, 36.2163086
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2943192, 29.2919540
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4565582, 32.4539566
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2674637, 26.2655411
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3500900, 23.3420563
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6096878, 16.6027279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1549

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1324

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5239243, upper bound: 13.4218558
time: 31.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5240000, upper bound: 13.4217300
time: 28.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7879562, 26.7834473
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5806808, 13.5794964
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3057785, 14.3027153
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5967789, 19.5924110
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5807800, 19.5815163
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1077271, 23.1035919
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2329483, 21.2266312
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1432037, 21.1439247
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8499832, 20.8470268
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4203262, 26.4186401
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7406158, 23.7328110
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5346451, 18.5421333
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6722031, 28.6629410
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1048965, 30.0965347
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9979553, 38.0053253
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6137810, 18.6130600
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7815094, 36.7837982
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1408615, 21.1518021
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9564705, 14.9613533
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9401016, 17.9474487
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6085968, 18.6181755
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3920326, 15.3973026
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5681381, 19.5720177
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7065544, 18.7184753
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4779282, 20.4855156
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7524414, 24.7653275
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3160248, 20.3297844
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8057861, 20.8143425
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1736412, 16.1769676
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7601509, 21.7774773
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0477066, 19.0531731
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1292992, 18.1167107
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3616257, 31.3608055
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8588181, 19.8563652
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8486176, 21.8502769
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5929680, 23.5915909
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2175903, 36.2159805
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2957764, 29.2905045
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4562607, 32.4542465
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2676239, 26.2653885
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3502350, 23.3419037
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6110039, 16.6014137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1714

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 685

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5213186, upper bound: 13.3897712
time: 41.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4880130, upper bound: 13.4231066
time: 31.19 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 74.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4280491, upper bound: 13.5100131
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4289960, upper bound: 13.5090742
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4330241, upper bound: 13.5224547
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4330241, upper bound: 13.5224547
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4420162, upper bound: 13.4969684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4188459, upper bound: 13.5201286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4420172, upper bound: 13.5206354
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4426829, upper bound: 13.5199624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4276382, upper bound: 13.5239756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4276382, upper bound: 13.5239756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4312635, upper bound: 13.5227624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4322665, upper bound: 13.5217050
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4169280, upper bound: 13.5157832
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4267220, upper bound: 13.5060751
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4223608, upper bound: 13.5141103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4214971, upper bound: 13.5149854
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.5215378, upper bound: 13.4162350
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4933930, upper bound: 13.4324770
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4967481, upper bound: 13.4319391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.5219300, upper bound: 13.4067603
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.5239243, upper bound: 13.4218558
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.5240000, upper bound: 13.4217300
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.5213186, upper bound: 13.3897712
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 74.61
Output dim: 12, lower bound: -13.4880130, upper bound: 13.4231066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7872925, 26.7718124
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5754509, 13.5688057
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2972260, 14.2919350
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5862579, 19.5842400
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5915375, 19.5836716
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0842209, 23.0803986
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2165451, 21.2291679
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1226730, 21.1146545
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8439484, 20.8332138
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4127884, 26.4079971
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7184296, 23.7173538
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5521736, 18.5474510
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6606522, 28.6724854
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0997849, 30.1019974
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9698792, 37.9573441
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6049881, 18.5994606
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7516327, 36.7474442
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1420670, 21.1379395
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9584579, 14.9566765
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9465561, 17.9463234
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6157761, 18.6138840
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3976669, 15.3981724
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5851021, 19.5804024
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7150345, 18.7085228
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4871063, 20.4868965
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7425537, 24.7362442
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3203125, 20.3115692
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8253708, 20.8269196
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1825752, 16.1830215
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7867203, 21.7833405
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0457382, 19.0452003
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1270790, 18.1386528
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3415451, 31.3484306
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8281822, 19.8362045
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8059158, 21.8132820
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5632858, 23.5699654
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2186584, 36.2170181
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2870712, 29.2905579
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4694443, 32.4710770
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2770157, 26.2775116
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3414841, 23.3477287
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6058350, 16.6107883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 947

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1635

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4323003, upper bound: 13.5221413
time: 20.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4327101, upper bound: 13.5217419
time: 26.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7873154, 26.7717896
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5762215, 13.5680351
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2969437, 14.2922173
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5858917, 19.5846062
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5924759, 19.5827408
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0844116, 23.0802078
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2162552, 21.2294464
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1231842, 21.1141434
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8445129, 20.8326492
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4133759, 26.4074097
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7185440, 23.7172394
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5520897, 18.5475349
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6608963, 28.6722260
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0998306, 30.1019478
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9726410, 37.9545822
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6056137, 18.5988274
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7529755, 36.7461014
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1420288, 21.1379776
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9585533, 14.9565811
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9469833, 17.9458961
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6167221, 18.6129379
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3987808, 15.3970585
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5838509, 19.5816612
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7150192, 18.7085381
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4876709, 20.4863205
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7425461, 24.7362518
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3203049, 20.3115730
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8252029, 20.8270950
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1836281, 16.1819687
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7874146, 21.7826538
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0460739, 19.0448723
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1264000, 18.1393356
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3400421, 31.3499298
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8261681, 19.8382149
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8046951, 21.8144989
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5618591, 23.5713959
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2167206, 36.2189560
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2857285, 29.2919006
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4679947, 32.4725266
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2751236, 26.2793999
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3393784, 23.3498383
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6041870, 16.6124344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1771

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4322816, upper bound: 13.5058806
time: 31.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4164723, upper bound: 13.5217115
time: 33.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7921524, 26.7882462
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5860939, 13.5833759
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2864113, 14.2847252
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5549927, 19.5582390
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5812988, 19.5756721
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0554352, 23.0584946
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2033768, 21.2171936
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1146927, 21.1128922
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8318710, 20.8323135
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4077911, 26.4062576
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7124405, 23.7185555
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5432510, 18.5366440
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6486359, 28.6625671
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0857315, 30.0907021
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9657288, 37.9613342
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6083374, 18.6077003
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7318115, 36.7331238
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1437302, 21.1334457
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9576836, 14.9516106
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9616318, 17.9553490
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6294403, 18.6221886
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4060421, 15.4027252
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5801086, 19.5748291
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7192764, 18.7081032
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4906998, 20.4851074
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7491760, 24.7372437
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3124695, 20.3024864
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8297539, 20.8230247
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1827583, 16.1819305
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8060112, 21.7968941
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0513725, 19.0471497
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1156235, 18.1297073
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3456192, 31.3492470
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8441429, 19.8475037
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8299522, 21.8264885
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5745354, 23.5732651
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.1969147, 36.1985168
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2895355, 29.2897530
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4707184, 32.4725571
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2513733, 26.2582245
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3269730, 23.3349457
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5804100, 16.5949974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4105384, upper bound: 13.5193425
time: 31.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4180586, upper bound: 13.5118216
time: 66.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8056488, 26.8043671
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5807762, 13.5798817
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3034897, 14.3029442
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5805931, 19.5751991
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5939102, 19.5917244
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0818939, 23.0785904
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2266693, 21.2288246
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1286469, 21.1257782
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8497772, 20.8461189
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4124374, 26.4111252
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7197418, 23.7252502
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5437813, 18.5417480
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6657486, 28.6793060
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1048508, 30.1107368
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9654999, 37.9618378
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6085014, 18.6080856
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7523346, 36.7527313
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1453705, 21.1380577
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9556160, 14.9537811
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9605179, 17.9540596
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6332474, 18.6252441
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4042187, 15.4012089
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5759735, 19.5722313
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7143402, 18.7066498
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4893723, 20.4829254
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7519264, 24.7407684
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3292542, 20.3186722
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8250427, 20.8200302
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1773529, 16.1785736
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8056717, 21.7944298
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0489693, 19.0463638
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1273232, 18.1380196
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3521500, 31.3505058
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8528748, 19.8481560
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8264809, 21.8224297
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5739288, 23.5710297
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2178192, 36.2180557
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2913208, 29.2878304
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4639816, 32.4641113
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2732086, 26.2741127
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3424301, 23.3467598
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6047211, 16.6094627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4270332, upper bound: 13.5126428
time: 29.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4338928, upper bound: 13.5057124
time: 31.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8062973, 26.8037186
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5814629, 13.5791950
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3032303, 14.3032074
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5800438, 19.5757408
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5939102, 19.5917244
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0818024, 23.0786819
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2261658, 21.2293358
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1287308, 21.1256943
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8497849, 20.8461151
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4130020, 26.4105682
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7201996, 23.7247963
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5438271, 18.5416985
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6664963, 28.6785583
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1048660, 30.1107254
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9668274, 37.9605103
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6087074, 18.6078835
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7526245, 36.7524338
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1452332, 21.1381950
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9556198, 14.9537773
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9605484, 17.9540253
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6335526, 18.6249313
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4044857, 15.4009399
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5757599, 19.5724449
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7142792, 18.7067070
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4894485, 20.4828491
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7518730, 24.7408295
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3291473, 20.3187790
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8249512, 20.8201180
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1780167, 16.1779099
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8056870, 21.7944221
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0489082, 19.0464249
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1270103, 18.1383362
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3509750, 31.3516769
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8515015, 19.8495293
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8255501, 21.8233604
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5729523, 23.5720024
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2172394, 36.2186356
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2907257, 29.2884293
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4626846, 32.4654083
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2721024, 26.2752228
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3416824, 23.3475037
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6042633, 16.6099205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4377477, upper bound: 13.5053693
time: 27.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4281017, upper bound: 13.5150183
time: 27.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8041992, 26.8026886
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5825615, 13.5801983
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3003960, 14.2996407
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5967484, 19.5982323
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5981522, 19.5925713
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1027451, 23.1039886
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2379227, 21.2498283
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1340714, 21.1294594
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8578491, 20.8535652
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4193268, 26.4181824
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7203674, 23.7275581
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5467606, 18.5392418
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6473846, 28.6633224
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0987930, 30.1079102
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9864960, 37.9743729
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6210136, 18.6185570
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7620850, 36.7596970
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1478806, 21.1354103
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9620132, 14.9560738
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9498024, 17.9418335
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6257515, 18.6150475
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4057465, 15.4001045
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5825577, 19.5761337
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7172508, 18.7032967
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4758453, 20.4674759
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7611008, 24.7457886
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3201714, 20.3043556
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8181877, 20.8109322
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1801834, 16.1777611
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7888832, 21.7720528
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0410690, 19.0359726
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1043625, 18.1217957
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3528442, 31.3555450
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8429489, 19.8475304
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8383980, 21.8380737
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5881424, 23.5903091
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2118683, 36.2111969
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3093567, 29.3129539
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4682922, 32.4705734
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2595139, 26.2620010
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3258057, 23.3362808
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5827942, 16.5957546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1593

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4012265, upper bound: 13.5171899
time: 32.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4208562, upper bound: 13.4975641
time: 27.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8031464, 26.8032913
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5821762, 13.5804119
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3003082, 14.2996826
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5963974, 19.5983124
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5983658, 19.5927773
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1026993, 23.1040039
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2380447, 21.2498474
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1338730, 21.1295052
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8573837, 20.8537979
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4190063, 26.4183807
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7203369, 23.7276115
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5465698, 18.5393028
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6475067, 28.6629028
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0988541, 30.1076775
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9865265, 37.9742508
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6207848, 18.6186447
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7621918, 36.7592163
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1478958, 21.1354332
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9621925, 14.9559669
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9498177, 17.9418182
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6258965, 18.6148453
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4060287, 15.3996849
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5821533, 19.5764389
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7172050, 18.7033081
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4758682, 20.4674187
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7609787, 24.7458344
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3200188, 20.3043900
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8181190, 20.8109856
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1802902, 16.1775856
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7889137, 21.7720604
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0414658, 19.0353584
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1044159, 18.1216507
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3526993, 31.3556747
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8427277, 19.8478165
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8383904, 21.8380814
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5880203, 23.5904007
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2109375, 36.2116928
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3086548, 29.3134003
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4683304, 32.4707336
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2589111, 26.2623711
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3253632, 23.3365936
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5819321, 16.5963497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4175403, upper bound: 13.5228889
time: 31.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4265492, upper bound: 13.5106303
time: 12.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7942123, 26.7884903
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5752449, 13.5710468
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2931595, 14.2888756
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5880280, 19.5847435
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5807648, 19.5704079
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0864563, 23.0814209
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2413406, 21.2529602
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1237946, 21.1164131
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8379440, 20.8271751
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4142532, 26.4115448
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7052078, 23.7062111
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5236168, 18.5224648
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6559753, 28.6697540
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0876846, 30.0903931
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9708862, 37.9572067
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6100922, 18.6048012
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7505951, 36.7448578
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1206131, 21.1153870
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9534111, 14.9500542
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9363823, 17.9329147
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6104393, 18.6049652
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4036446, 15.3999596
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5662003, 19.5655403
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6925201, 18.6853714
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4718437, 20.4684715
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7330322, 24.7252579
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.2944183, 20.2869110
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.7947617, 20.7953186
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1781693, 16.1777649
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7582397, 21.7518921
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0393639, 19.0352974
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1158676, 18.1283913
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3399734, 31.3448181
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8207550, 19.8290520
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8138504, 21.8173752
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5708466, 23.5757065
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.1918945, 36.1951599
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3005829, 29.3053703
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4699097, 32.4706650
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2571793, 26.2599754
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3215942, 23.3312912
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5849228, 16.5963879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1731

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4223400, upper bound: 13.5215122
time: 36.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4303297, upper bound: 13.5162880
time: 42.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7921829, 26.7905197
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5739212, 13.5723686
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2915039, 14.2905312
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5860672, 19.5867043
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5771332, 19.5740318
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0830994, 23.0847855
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2413559, 21.2529411
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1218338, 21.1183701
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8339844, 20.8311348
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4133911, 26.4124069
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7019653, 23.7094574
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5271873, 18.5188942
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6542969, 28.6714401
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0845184, 30.0935593
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9685516, 37.9595413
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6078339, 18.6070595
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7480774, 36.7473755
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1247482, 21.1112556
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9545441, 14.9489174
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9385262, 17.9307747
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6127892, 18.6026192
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4038658, 15.3997402
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5702133, 19.5615196
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6963043, 18.6815872
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4740868, 20.4662323
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7374191, 24.7208710
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.2982941, 20.2830353
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.7993851, 20.7906914
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1784363, 16.1774940
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7632904, 21.7468491
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0395622, 19.0351028
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1133652, 18.1309013
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3419647, 31.3428268
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8240814, 19.8257256
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8174210, 21.8138084
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5734100, 23.5731468
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.1948395, 36.1922150
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3017883, 29.3041611
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4700012, 32.4705658
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2574158, 26.2597389
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3221130, 23.3307724
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5846672, 16.5966434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4251758, upper bound: 13.5213263
time: 31.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4318924, upper bound: 13.5147068
time: 30.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7936325, 26.7964325
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5768280, 13.5791588
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2984543, 14.2964287
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5992661, 19.6003761
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5574303, 19.5576439
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1131287, 23.1133423
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2289429, 21.2192230
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1463928, 21.1509285
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8316078, 20.8344116
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4238052, 26.4264374
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7388687, 23.7331085
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5166435, 18.5266876
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6882782, 28.6729279
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0910950, 30.0825500
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9924164, 38.0063248
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6099930, 18.6111221
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7758331, 36.7818069
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1275787, 21.1410332
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9535332, 14.9564629
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9331207, 17.9414940
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.5931244, 18.6058655
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3970642, 15.4017029
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5380440, 19.5461960
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6890144, 18.7017174
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4689255, 20.4777069
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7273712, 24.7445831
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3038940, 20.3178291
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.7866211, 20.7967186
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1783485, 16.1808128
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7196083, 21.7418861
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0534058, 19.0560875
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1292801, 18.1112061
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3660202, 31.3623047
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8285065, 19.8250275
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8546906, 21.8559532
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5888672, 23.5849533
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2196732, 36.2185211
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2761993, 29.2712631
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4396973, 32.4326324
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2624359, 26.2568054
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3421936, 23.3304443
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6088791, 16.5993156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1409

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 521

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5113058, upper bound: 13.4158239
time: 30.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5211237, upper bound: 13.4060082
time: 29.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7590103, 26.7667999
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5635529, 13.5680103
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2778244, 14.2832603
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5966568, 19.5980911
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5496559, 19.5590324
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0983963, 23.1027908
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.1979370, 21.1818199
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1259003, 21.1363564
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8107796, 20.8196678
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4178085, 26.4203644
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7347260, 23.7289543
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5372734, 18.5442963
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6680145, 28.6462860
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1008224, 30.0948181
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9887085, 38.0027313
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5990601, 18.6012421
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7815247, 36.7895584
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1376648, 21.1492310
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9469833, 14.9516296
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9406738, 17.9477577
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6085434, 18.6163521
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3933678, 15.3965130
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5591545, 19.5627899
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6927719, 18.7072601
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4772415, 20.4830170
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7356415, 24.7497406
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3099976, 20.3242569
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8053741, 20.8119850
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1752281, 16.1754341
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7327805, 21.7449379
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0430527, 19.0455284
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1154709, 18.0937080
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3520355, 31.3426437
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8364182, 19.8244438
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8359833, 21.8297005
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5840912, 23.5785179
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2047577, 36.1992874
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2690125, 29.2639084
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4391174, 32.4331741
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2595062, 26.2534752
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3387070, 23.3262482
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6037331, 16.5890255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1011

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5213781, upper bound: 13.4020389
time: 41.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5172039, upper bound: 13.4062086
time: 27.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7844925, 26.7857590
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5796585, 13.5807304
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3044701, 14.3042488
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5933914, 19.5960083
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5802727, 19.5816193
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1041107, 23.1062622
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2322083, 21.2262650
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1412354, 21.1454124
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8465004, 20.8513145
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4188690, 26.4197998
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7394943, 23.7341614
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5362244, 18.5425758
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6741180, 28.6609421
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1046524, 30.0974655
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9970398, 38.0066681
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6134415, 18.6143837
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7809143, 36.7843933
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1419449, 21.1518135
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9574623, 14.9601059
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9399071, 17.9471626
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6088104, 18.6181602
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3919563, 15.3959389
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5677032, 19.5720787
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7076454, 18.7180328
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4779930, 20.4851837
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7529221, 24.7656403
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3166885, 20.3297997
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8072052, 20.8134613
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1745567, 16.1756287
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7608795, 21.7778244
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0476379, 19.0512085
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1295776, 18.1166496
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3614731, 31.3607559
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8574791, 19.8568459
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8484573, 21.8500786
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5927887, 23.5913734
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2160950, 36.2149124
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2942429, 29.2918472
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4570084, 32.4543839
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2670898, 26.2650909
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3495255, 23.3413734
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6089439, 16.6018372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5168627, upper bound: 13.4097361
time: 27.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5116854, upper bound: 13.4147911
time: 34.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7843781, 26.7858734
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5796547, 13.5807323
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3044739, 14.3042526
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5933609, 19.5960426
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5805168, 19.5813141
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1041794, 23.1062012
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2322464, 21.2262192
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1412735, 21.1453857
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8465614, 20.8512764
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4188461, 26.4198227
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7394791, 23.7341690
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5360107, 18.5428200
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6741486, 28.6609116
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1046753, 30.0974503
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9971161, 38.0065918
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6134720, 18.6143379
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7810364, 36.7842789
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1419144, 21.1518440
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9574852, 14.9600830
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9399300, 17.9471359
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6088104, 18.6181526
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3920860, 15.3958092
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5674667, 19.5723152
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7076302, 18.7180481
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4780235, 20.4851646
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7528992, 24.7656708
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3166962, 20.3297882
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8071289, 20.8135452
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1745949, 16.1755905
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7609406, 21.7777672
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0477448, 19.0511017
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1295547, 18.1166611
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3614426, 31.3607826
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8574028, 19.8569298
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8484116, 21.8501167
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5927582, 23.5914001
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2158661, 36.2151489
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2942200, 29.2918777
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4569931, 32.4544067
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2670135, 26.2651596
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3494034, 23.3414917
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6087990, 16.6019821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1791

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5228522, upper bound: 13.4079545
time: 39.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5102382, upper bound: 13.4205735
time: 27.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7817841, 26.7773590
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5798416, 13.5799255
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3047447, 14.3019638
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5930595, 19.5845566
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5792770, 19.5821991
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1057281, 23.1006012
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2328949, 21.2261887
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1411667, 21.1428070
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8490982, 20.8466835
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4159622, 26.4136658
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7426987, 23.7254410
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5314484, 18.5398865
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6609268, 28.6452789
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0955963, 30.0819588
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9913635, 38.0005951
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6128845, 18.6129227
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7718658, 36.7686844
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1257172, 21.1427536
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9455032, 14.9545860
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9214401, 17.9353676
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.5997162, 18.6125107
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3865700, 15.3941612
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5679626, 19.5719109
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6913338, 18.7087593
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4661407, 20.4779968
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7357941, 24.7553558
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.2868423, 20.3111610
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.7936249, 20.8062782
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1715736, 16.1755409
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7581367, 21.7758904
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0318375, 19.0431786
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1228104, 18.1065445
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3502731, 31.3440742
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8402786, 19.8274040
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8365479, 21.8359680
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5919380, 23.5933304
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2026825, 36.2003708
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2908173, 29.2900009
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4528961, 32.4504623
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2613907, 26.2556305
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3451843, 23.3339844
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6103439, 16.5961914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1787

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4926874, upper bound: 13.3870792
time: 32.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5186217, upper bound: 13.3611498
time: 26.41 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 60.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4323003, upper bound: 13.5221413
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4327101, upper bound: 13.5217419
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4322816, upper bound: 13.5058806
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4164723, upper bound: 13.5217115
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4105384, upper bound: 13.5193425
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4180586, upper bound: 13.5118216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4270332, upper bound: 13.5126428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4338928, upper bound: 13.5057124
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4377477, upper bound: 13.5053693
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4281017, upper bound: 13.5150183
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4012265, upper bound: 13.5171899
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4208562, upper bound: 13.4975641
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4175403, upper bound: 13.5228889
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4265492, upper bound: 13.5106303
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4223400, upper bound: 13.5215122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4303297, upper bound: 13.5162880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4251758, upper bound: 13.5213263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4318924, upper bound: 13.5147068
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5113058, upper bound: 13.4158239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5211237, upper bound: 13.4060082
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5213781, upper bound: 13.4020389
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5172039, upper bound: 13.4062086
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5168627, upper bound: 13.4097361
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5116854, upper bound: 13.4147911
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5228522, upper bound: 13.4079545
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5102382, upper bound: 13.4205735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.4926874, upper bound: 13.3870792
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 60.71
Output dim: 12, lower bound: -13.5186217, upper bound: 13.3611498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7682495, 26.7429047
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5674515, 13.5568867
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2868614, 14.2766037
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5723343, 19.5633583
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5816917, 19.5688972
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0692139, 23.0571747
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2048569, 21.2213364
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1146393, 21.1016197
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8283005, 20.8101196
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4028854, 26.3928070
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7044983, 23.6966019
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5426979, 18.5409698
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6592484, 28.6715469
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0838547, 30.0789108
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9573975, 37.9386444
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6005096, 18.5932846
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7433929, 36.7340851
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1273041, 21.1276627
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9507866, 14.9514465
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9352112, 17.9387627
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6095428, 18.6091461
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3871689, 15.3914146
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5772133, 19.5749664
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.6970215, 18.6965485
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4731979, 20.4775620
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7298355, 24.7279587
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.2954178, 20.2946739
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8030624, 20.8117981
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1738014, 16.1772156
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7616501, 21.7663765
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0360794, 19.0385361
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1257439, 18.1377754
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3213654, 31.3348465
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8076324, 19.8225174
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.7902832, 21.8026924
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5511398, 23.5619774
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2032318, 36.2063751
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2852707, 29.2892838
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4677505, 32.4707794
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2632446, 26.2692413
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3330612, 23.3422165
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6053505, 16.6103401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 802

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4287816, upper bound: 13.5217054
time: 35.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4319014, upper bound: 13.5186177
time: 29.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7583847, 26.7527771
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5635300, 13.5608063
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2818947, 14.2815704
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5653763, 19.5703125
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5767632, 19.5738297
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0609894, 23.0653915
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2087021, 21.2174873
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1096420, 21.1066170
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8208542, 20.8175659
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.3976059, 26.3980865
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.6976776, 23.7034302
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5456886, 18.5379753
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6597061, 28.6710892
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0766983, 30.0860672
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9511871, 37.9448624
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.5988083, 18.5949821
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7382660, 36.7392120
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1317902, 21.1231689
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9532242, 14.9490089
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9389954, 17.9349823
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6110382, 18.6076508
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3909111, 15.3876724
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5796700, 19.5725174
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7030640, 18.6905136
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4777756, 20.4729881
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7342606, 24.7235413
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3034134, 20.2866821
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8102493, 20.8046112
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1767693, 16.1742477
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7697525, 21.7582664
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0390701, 19.0355377
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1262016, 18.1373138
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3279572, 31.3282471
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8144913, 19.8156586
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.7953186, 21.7976532
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5552979, 23.5578194
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2080078, 36.2015991
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2858047, 29.2887497
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4691391, 32.4693832
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2687378, 26.2637482
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3359680, 23.3393097
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6053848, 16.6103058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1393

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1010

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4318501, upper bound: 13.5212181
time: 32.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4321882, upper bound: 13.5208819
time: 25.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7753067, 26.7574310
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5716896, 13.5626163
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2955360, 14.2905769
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5798531, 19.5774231
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5957909, 19.5860901
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0747910, 23.0688400
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2189407, 21.2320099
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1131134, 21.1021385
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8407288, 20.8282280
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4067993, 26.3995743
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7189331, 23.7176056
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5515862, 18.5468025
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6539917, 28.6660538
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0991745, 30.1012115
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9567261, 37.9356613
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6077576, 18.6013870
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7391815, 36.7295609
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1435738, 21.1394806
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9506454, 14.9499855
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9490204, 17.9477310
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6172409, 18.6134567
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.3929138, 15.3920593
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5847855, 19.5826035
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7143517, 18.7078819
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4877739, 20.4864388
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7435150, 24.7371979
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3221893, 20.3134766
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8251686, 20.8270645
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1836624, 16.1819954
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7867203, 21.7812424
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0353508, 19.0358849
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1160202, 18.1302681
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3267899, 31.3388176
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8163719, 19.8299980
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.7897339, 21.8019600
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5521164, 23.5632248
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2062225, 36.2101593
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2764587, 29.2841263
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4449997, 32.4525146
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2695618, 26.2747345
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3276367, 23.3399925
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6068153, 16.6149979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 786

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 978

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4153585, upper bound: 13.5216277
time: 33.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4163833, upper bound: 13.5205571
time: 26.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.7906342, 26.7862320
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5858879, 13.5830421
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.2842979, 14.2821159
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5544510, 19.5575333
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5810394, 19.5751457
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.0540695, 23.0568161
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2032928, 21.2171326
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1130371, 21.1108589
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8306351, 20.8306770
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4074936, 26.4058914
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7111893, 23.7170868
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5429840, 18.5363998
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6452026, 28.6598434
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0857086, 30.0906715
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9651031, 37.9601440
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6079102, 18.6071625
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7315979, 36.7328186
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1436005, 21.1332703
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9576187, 14.9515266
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9615021, 17.9552116
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6286163, 18.6212349
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4046154, 15.4008293
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5795135, 19.5743256
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7189941, 18.7077980
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4902344, 20.4847374
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7491455, 24.7371979
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3116760, 20.3015976
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8296013, 20.8229637
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1820984, 16.1815453
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8053551, 21.7962990
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0512733, 19.0470695
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1155624, 18.1296730
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3438416, 31.3480721
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8418961, 19.8456879
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8277016, 21.8247299
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5742569, 23.5731049
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.1963654, 36.1981049
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.2886124, 29.2890511
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4706879, 32.4724808
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2507858, 26.2579041
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3263474, 23.3345718
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5799484, 16.5947685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4101583, upper bound: 13.5087149
time: 26.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.3998577, upper bound: 13.5189644
time: 41.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8032074, 26.8033371
1: -10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5830040, 13.5810966
2: -14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3008270, 14.3001442
3: -21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.5972862, 19.5992088
4: -22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.6004868, 19.5944519
5: -20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1040268, 23.1053772
6: -22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2356415, 21.2478714
7: -21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1311646, 21.1265259
8: -34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8552208, 20.8511772
9: -12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4181824, 26.4175873
10: -6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7159958, 23.7241898
11: -6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5436020, 18.5358162
12: 0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.6344910, 28.6511307
13: -10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.0922470, 30.1021423
14: -33.1121063, 10.5235691, -33.1121063, 10.5235691, -37.9827423, 37.9693527
15: -20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6190910, 18.6166687
16: -14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293
17: -21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7624664, 36.7594070
18: -14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1480103, 21.1342163
19: -10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9593620, 14.9525070
20: -15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9421844, 17.9331207
21: -11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6190758, 18.6067963
22: -9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4022331, 15.3950768
23: -14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5803757, 19.5744286
24: -17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7072830, 18.6921234
25: -11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4708824, 20.4613800
26: -16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7663879, 24.7498322
27: -27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3070755, 20.2895393
28: -16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8116302, 20.8036308
29: -7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1782265, 16.1751595
30: -19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.7741699, 21.7554855
31: -13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0386887, 19.0320892
32: -12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.0969162, 18.1152725
33: -45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3547440, 31.3576355
34: -42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8429947, 19.8480034
35: -29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8406639, 21.8402138
36: -23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5883408, 23.5907974
37: -43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2115707, 36.2124176
38: -30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3085327, 29.3133163
39: -38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4683838, 32.4707870
40: -44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2589340, 26.2624054
41: -24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3189011, 23.3311539
42: -19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.5738945, 16.5893364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 50.30 + 3550.91 = 3601.22 seconds
