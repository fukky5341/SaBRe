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
execution time: IAR + RelationalAnalysis = 2.59 + 46.98 = 49.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -13.5325782, upper bound: 13.5325782

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5270205, upper bound: 13.4992576
time: 36.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5270205, upper bound: 13.5270203
time: 24.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 60.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 60.47
Output dim: 12, lower bound: -13.5270205, upper bound: 13.4992576
IS_A2, status: Status.UNKNOWN, split count: 1, time: 60.47
Output dim: 12, lower bound: -13.5270205, upper bound: 13.5270203

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.0930195, 4.0647273, -29.0994663, 4.0751143, -26.8002167, 26.7956696
1: -10.1970205, 6.6673579, -10.1989918, 6.6740618, -13.5878906, 13.5826225
2: -14.3185463, 4.4691973, -14.3227577, 4.4752302, -14.3029366, 14.3015099
3: -20.9902191, 0.7945735, -21.0008736, 0.7995965, -19.5804977, 19.5851822
4: -22.1459446, 3.3632917, -22.1542091, 3.3691134, -19.5716095, 19.5819016
5: -20.5381508, 5.7700996, -20.5443039, 5.7773743, -23.0834732, 23.0849304
6: -22.5271721, 3.2136421, -22.5358753, 3.2158098, -21.2402573, 21.2480011
7: -21.4347343, 4.0009413, -21.4421730, 4.0092416, -21.1277313, 21.1298943
8: -34.1496773, -4.0856628, -34.1528664, -4.0779920, -20.8740768, 20.8702927
9: -12.3104849, 16.6399193, -12.3166113, 16.6482315, -26.3966904, 26.4037628
10: -6.4137688, 20.6781254, -6.4196529, 20.6872234, -23.7165756, 23.7169495
11: -6.9492145, 13.9898539, -6.9576731, 13.9932747, -18.5550919, 18.5573273
12: 0.6997776, 35.2869415, 0.6905327, 35.3166008, -28.6642609, 28.6396484
13: -10.7427044, 24.2756920, -10.7481604, 24.2947636, -30.1063690, 30.0965118
14: -33.0887413, 10.4007969, -33.1029091, 10.4756927, -37.9414215, 37.8796844
15: -20.7264290, 0.3071406, -20.7317123, 0.3119676, -18.6093369, 18.6115913
16: -14.5467339, 7.4597969, -14.5519543, 7.4654779, -22.0122108, 22.0117512
17: -21.3129921, 18.7184887, -21.3244133, 18.7849731, -36.7172394, 36.6600723
18: -14.6977320, 9.5213490, -14.7136803, 9.5264931, -21.1307106, 21.1378670
19: -10.8461113, 6.8206267, -10.8522415, 6.8232422, -14.9561348, 14.9591408
20: -15.0731783, 4.9714751, -15.0810280, 4.9755344, -17.9515419, 17.9547234
21: -11.3693619, 9.6155043, -11.3790083, 9.6189642, -18.6332626, 18.6358910
22: -9.6150703, 7.8766537, -9.6205788, 7.8800073, -15.4011364, 15.4025536
23: -14.0568752, 7.3682270, -14.0636616, 7.3726764, -19.5830688, 19.5813828
24: -17.4700718, 6.2168870, -17.4772854, 6.2194200, -18.7329025, 18.7344666
25: -11.3668528, 10.1535749, -11.3728123, 10.1589985, -20.4830208, 20.4813271
26: -16.2036514, 9.8245411, -16.2138729, 9.8262863, -24.7438164, 24.7493057
27: -27.5327988, 0.8367038, -27.5475998, 0.8402677, -20.3363495, 20.3431931
28: -16.3267727, 7.3226671, -16.3328781, 7.3265557, -20.8252182, 20.8254890
29: -7.3737936, 10.3902922, -7.3789158, 10.4039259, -16.1730728, 16.1641121
30: -19.1695442, 7.4139385, -19.1772327, 7.4195776, -21.8173752, 21.8180351
31: -13.1080475, 9.5233765, -13.1167259, 9.5261974, -19.0406685, 19.0464668
32: -12.5323792, 9.3051357, -12.5420570, 9.3095045, -18.1381340, 18.1483383
33: -45.4287796, -9.3926029, -45.4582214, -9.3875523, -31.3308716, 31.3446732
34: -41.9648209, -14.0882702, -41.9908791, -14.0848913, -19.8399239, 19.8531914
35: -29.0430794, -2.4589570, -29.0532761, -2.4574716, -21.8334312, 21.8359680
36: -23.7307549, 3.7683632, -23.7384300, 3.7712932, -23.5808411, 23.5842056
37: -43.6354065, -4.6352091, -43.6638641, -4.6317987, -36.1842194, 36.1945190
38: -30.0036030, 1.3859138, -30.0127354, 1.3879960, -29.2944489, 29.2903595
39: -38.9275703, -3.9669569, -38.9480934, -3.9636559, -32.4506378, 32.4620819
40: -44.3141785, -12.4165859, -44.3545761, -12.4129381, -26.2255402, 26.2528381
41: -24.2625599, 5.0286598, -24.2851505, 5.0311780, -23.3401260, 23.3550797
42: -19.4725952, 2.2648211, -19.4809189, 2.2686653, -16.6130791, 16.6169472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1733

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4515586
time: 36.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4955447
time: 27.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.1159344, 4.1058183, -29.1010990, 4.0798645, -26.8277817, 26.8455582
1: -10.2226114, 6.6971731, -10.1993694, 6.6765013, -13.6166267, 13.6238518
2: -14.3259583, 4.4947577, -14.3228302, 4.4783335, -14.3145409, 14.3313751
3: -21.0091515, 0.8480990, -21.0054436, 0.8021920, -19.6003571, 19.6445389
4: -22.1556873, 3.4076028, -22.1545868, 3.3717051, -19.5928307, 19.6498528
5: -20.5442810, 5.8128476, -20.5442295, 5.7811174, -23.0936050, 23.1487350
6: -22.5528851, 3.2450247, -22.5351658, 3.2168226, -21.2706070, 21.2944679
7: -21.4565773, 4.0469594, -21.4425316, 4.0136070, -21.1563950, 21.1956406
8: -34.1637955, -4.0519543, -34.1539993, -4.0741367, -20.8989563, 20.9077110
9: -12.3288097, 16.6925602, -12.3188314, 16.6521473, -26.4109650, 26.4935760
10: -6.4375362, 20.7253723, -6.4226484, 20.6911201, -23.7317734, 23.7891235
11: -7.0239573, 14.0042152, -6.9605742, 13.9948549, -18.6438560, 18.5736046
12: 0.5844698, 35.3206825, 0.6857805, 35.3260078, -28.8192749, 28.6817017
13: -10.8616114, 24.3137169, -10.7506714, 24.3013954, -30.2385788, 30.1361923
14: -33.2910004, 10.5157127, -33.1100121, 10.5147038, -38.1965332, 37.9847717
15: -20.7451286, 0.3739209, -20.7338657, 0.3133001, -18.6434479, 18.6809425
16: -14.5708399, 7.5462103, -14.5530891, 7.4681492, -22.0389900, 22.0993004
17: -21.5002594, 18.8185711, -21.3303051, 18.8175278, -36.9425354, 36.7578430
18: -14.7329855, 9.5468969, -14.7207718, 9.5289803, -21.1766052, 21.1639328
19: -10.8816442, 6.8369107, -10.8553200, 6.8245144, -14.9972687, 14.9800606
20: -15.1067038, 5.0002170, -15.0836029, 4.9776382, -17.9885406, 17.9923553
21: -11.4152117, 9.6275997, -11.3838463, 9.6203384, -18.6806030, 18.6673813
22: -9.6709299, 7.9114847, -9.6231575, 7.8799119, -15.4576931, 15.4451408
23: -14.0868721, 7.4005795, -14.0662775, 7.3749628, -19.6478310, 19.6054955
24: -17.4930286, 6.2444444, -17.4801216, 6.2204766, -18.7788467, 18.7562714
25: -11.4255066, 10.1836872, -11.3758583, 10.1588001, -20.5542679, 20.5130730
26: -16.2815971, 9.8372850, -16.2186661, 9.8267536, -24.8476067, 24.7585526
27: -27.5512009, 0.8897152, -27.5504570, 0.8420353, -20.3630753, 20.4089355
28: -16.3626938, 7.3511305, -16.3355103, 7.3284903, -20.8888588, 20.8491020
29: -7.4466095, 10.4207344, -7.3812332, 10.4108467, -16.2531357, 16.1931267
30: -19.2015781, 7.4465017, -19.1808376, 7.4221926, -21.8538017, 21.8548584
31: -13.1461830, 9.5478554, -13.1211624, 9.5273418, -19.0849419, 19.0792084
32: -12.5565710, 9.3412828, -12.5405436, 9.3117485, -18.1729355, 18.2088165
33: -45.5057449, -9.2795534, -45.4734879, -9.3866234, -31.4107132, 31.4747734
34: -42.0038910, -13.9870510, -42.0009842, -14.0838461, -19.8767624, 19.9681816
35: -29.0792007, -2.4262350, -29.0567513, -2.4581640, -21.9015350, 21.8523788
36: -23.7706795, 3.8118658, -23.7387238, 3.7727580, -23.6603928, 23.6013947
37: -43.7095184, -4.5761924, -43.6780777, -4.6302528, -36.2709579, 36.2235641
38: -30.0515347, 1.4034839, -30.0159416, 1.3889904, -29.3908386, 29.2824249
39: -38.9878311, -3.8882139, -38.9573898, -3.9636354, -32.5249252, 32.5290833
40: -44.3912125, -12.3172045, -44.3751755, -12.4112072, -26.2920685, 26.3773880
41: -24.3059139, 5.1107931, -24.2967472, 5.0324464, -23.3730621, 23.4475555
42: -19.4982185, 2.2938933, -19.4814301, 2.2704983, -16.6571732, 16.6556435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1733

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4793358
time: 44.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5233240, upper bound: 13.5233236
time: 42.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 89.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 89.45
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4515586
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 89.45
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4955447
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 89.45
Output dim: 12, lower bound: -13.5233240, upper bound: 13.4793358
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 89.45
Output dim: 12, lower bound: -13.5233240, upper bound: 13.5233236

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -29.0907364, 4.0634880, -29.0937595, 4.0720863, -26.7951355, 26.7885208
1: -10.1951542, 6.6668787, -10.1942434, 6.6728621, -13.5842514, 13.5760479
2: -14.3176746, 4.4686947, -14.3205309, 4.4739017, -14.3007622, 14.2989159
3: -20.9892292, 0.7924626, -20.9983845, 0.7943788, -19.5692139, 19.5780602
4: -22.1447983, 3.3623705, -22.1512890, 3.3669004, -19.5660400, 19.5738220
5: -20.5372810, 5.7658844, -20.5421600, 5.7670498, -23.0651855, 23.0757980
6: -22.5197754, 3.2129459, -22.5174198, 3.2140446, -21.2300491, 21.2307587
7: -21.4330063, 3.9965506, -21.4379196, 3.9983740, -21.1137238, 21.1205902
8: -34.1487808, -4.0874305, -34.1505814, -4.0823793, -20.8688698, 20.8653526
9: -12.3054752, 16.6387863, -12.3041077, 16.6454315, -26.3882294, 26.3888321
10: -6.4102097, 20.6762657, -6.4107947, 20.6825886, -23.7070923, 23.7057114
11: -6.9471927, 13.9811430, -6.9526963, 13.9724197, -18.5307961, 18.5431900
12: 0.7178106, 35.2861481, 0.7359991, 35.3144646, -28.6442490, 28.5938339
13: -10.7312279, 24.2750072, -10.7192402, 24.2930641, -30.0930939, 30.0671310
14: -33.0867844, 10.3945560, -33.0980377, 10.4600859, -37.9280243, 37.8697510
15: -20.7255459, 0.3015366, -20.7294369, 0.2979324, -18.5960693, 18.6032562
16: -14.5418024, 7.4588447, -14.5396271, 7.4630947, -22.0048981, 21.9984722
17: -21.3103561, 18.7174034, -21.3178692, 18.7823410, -36.7097015, 36.6484070
18: -14.6970119, 9.5137806, -14.7119427, 9.5074253, -21.1152000, 21.1297607
19: -10.8452110, 6.8142891, -10.8499746, 6.8075066, -14.9396858, 14.9505653
20: -15.0720291, 4.9619908, -15.0781336, 4.9516435, -17.9267387, 17.9425125
21: -11.3683624, 9.6074753, -11.3765202, 9.5987129, -18.6109085, 18.6251106
22: -9.6144581, 7.8685503, -9.6189842, 7.8596835, -15.3797874, 15.3928585
23: -14.0559702, 7.3596487, -14.0613747, 7.3513370, -19.5606880, 19.5708961
24: -17.4697647, 6.2043977, -17.4765072, 6.1878810, -18.7011528, 18.7212296
25: -11.3662672, 10.1428680, -11.3713474, 10.1320791, -20.4559860, 20.4691429
26: -16.2026939, 9.8174734, -16.2114658, 9.8086071, -24.7284889, 24.7392807
27: -27.5321465, 0.8231211, -27.5460625, 0.8059278, -20.3017960, 20.3280716
28: -16.3253880, 7.3142896, -16.3294430, 7.3053961, -20.8027534, 20.8140106
29: -7.3732367, 10.3844700, -7.3774939, 10.3892450, -16.1574821, 16.1568642
30: -19.1684608, 7.4005880, -19.1746082, 7.3861198, -21.7817764, 21.8018875
31: -13.1071167, 9.5136929, -13.1142807, 9.5020981, -19.0159225, 19.0344276
32: -12.5220604, 9.3047762, -12.5170107, 9.3086052, -18.1273193, 18.1248665
33: -45.4223404, -9.3947840, -45.4424095, -9.3929005, -31.3173523, 31.3210373
34: -41.9644165, -14.0892153, -41.9899216, -14.0873632, -19.8367195, 19.8509560
35: -29.0416660, -2.4599741, -29.0497398, -2.4599969, -21.8276672, 21.8272133
36: -23.7294312, 3.7679553, -23.7351341, 3.7702641, -23.5781898, 23.5795135
37: -43.6267509, -4.6366491, -43.6421509, -4.6354661, -36.1687927, 36.1636581
38: -30.0024414, 1.3849874, -30.0099106, 1.3857460, -29.2899780, 29.2842674
39: -38.9184494, -3.9685822, -38.9257507, -3.9676816, -32.4363403, 32.4327850
40: -44.3045807, -12.4170351, -44.3309975, -12.4140129, -26.2134628, 26.2251663
41: -24.2533455, 5.0280142, -24.2619781, 5.0296292, -23.3287277, 23.3307343
42: -19.4629650, 2.2639542, -19.4575539, 2.2664900, -16.6011963, 16.5918846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1706

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4460581
time: 38.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5208538, upper bound: 13.4487189
time: 37.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -29.0921040, 4.0640273, -29.1111164, 4.0853324, -26.8133087, 26.8049622
1: -10.1959991, 6.6670218, -10.2011719, 6.6841736, -13.6002960, 13.5819263
2: -14.3182812, 4.4685578, -14.3334789, 4.4809113, -14.3100281, 14.3139343
3: -20.9899254, 0.7928009, -21.0253448, 0.8074872, -19.5844078, 19.6160049
4: -22.1455879, 3.3624692, -22.1642551, 3.3726830, -19.6002655, 19.5797119
5: -20.5379162, 5.7676878, -20.5852413, 5.7874718, -23.0893555, 23.1297607
6: -22.5237827, 3.2133560, -22.5384808, 3.2631130, -21.2807236, 21.2477798
7: -21.4341202, 3.9986765, -21.4744911, 4.0109615, -21.1270447, 21.1608582
8: -34.1493683, -4.0883608, -34.1673012, -4.0803156, -20.8788185, 20.8785515
9: -12.3076820, 16.6394138, -12.3231668, 16.6891193, -26.4386673, 26.4105225
10: -6.4108663, 20.6774826, -6.4160433, 20.7285023, -23.7463608, 23.7163620
11: -6.9483714, 13.9880123, -7.0001974, 13.9938030, -18.5492973, 18.6038666
12: 0.7037773, 35.2867203, 0.6909771, 35.4267197, -28.7722473, 28.6316071
13: -10.7401896, 24.2753887, -10.7498999, 24.3562870, -30.1640091, 30.0925674
14: -33.0879822, 10.3959455, -33.1598282, 10.4744711, -37.9422913, 37.9291687
15: -20.7259960, 0.3057694, -20.7528610, 0.3143792, -18.6069756, 18.6294136
16: -14.5441971, 7.4592657, -14.5550613, 7.4848013, -22.0289993, 22.0143280
17: -21.3115997, 18.7180576, -21.3309975, 18.8115063, -36.7471619, 36.6578522
18: -14.6971769, 9.5191441, -14.7585135, 9.5280256, -21.1330795, 21.1738739
19: -10.8458309, 6.8193278, -10.8904247, 6.8212929, -14.9513321, 14.9994736
20: -15.0727596, 4.9694548, -15.1341429, 4.9756250, -17.9480782, 18.0082664
21: -11.3690281, 9.6128244, -11.4272652, 9.6170034, -18.6263618, 18.6837578
22: -9.6146259, 7.8742905, -9.6586943, 7.8801699, -15.3969288, 15.4399719
23: -14.0566158, 7.3663378, -14.1161699, 7.3735504, -19.5796738, 19.6343765
24: -17.4699173, 6.2143984, -17.5433388, 6.2192411, -18.7253036, 18.7989349
25: -11.3666477, 10.1507511, -11.4319258, 10.1621056, -20.4818954, 20.5415459
26: -16.2029629, 9.8230400, -16.2566013, 9.8273468, -24.7482300, 24.7694321
27: -27.5325966, 0.8340521, -27.6296978, 0.8378253, -20.3251419, 20.4243851
28: -16.3263836, 7.3206906, -16.3928070, 7.3251753, -20.8199158, 20.8896561
29: -7.3734908, 10.3885612, -7.4200063, 10.4041252, -16.1715431, 16.2085381
30: -19.1690979, 7.4110746, -19.2436295, 7.4198246, -21.8084564, 21.8874359
31: -13.1076775, 9.5212889, -13.1672087, 9.5249109, -19.0357208, 19.0995522
32: -12.5301800, 9.3049393, -12.5489254, 9.3699760, -18.1974945, 18.1489639
33: -45.4263611, -9.3933744, -45.4681969, -9.3315029, -31.3939209, 31.3471756
34: -41.9646568, -14.0885620, -41.9996109, -14.0814915, -19.8468170, 19.8641815
35: -29.0418968, -2.4592841, -29.0626411, -2.4495933, -21.8538971, 21.8377151
36: -23.7294655, 3.7682118, -23.7495651, 3.7822866, -23.5926552, 23.5956612
37: -43.6330643, -4.6356502, -43.6780472, -4.5676537, -36.2546005, 36.1988297
38: -30.0024071, 1.3857179, -30.0187645, 1.4057312, -29.3134155, 29.3016281
39: -38.9243469, -3.9674790, -38.9548798, -3.8986259, -32.5237808, 32.4612961
40: -44.3117790, -12.4167719, -44.3641243, -12.3569241, -26.2810669, 26.2576218
41: -24.2591019, 5.0284691, -24.2933121, 5.0937109, -23.4051590, 23.3587799
42: -19.4702816, 2.2644539, -19.4815121, 2.3230143, -16.6662598, 16.6103287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1706

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4900419
time: 29.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4927048
time: 32.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.1136608, 4.1046925, -29.0954189, 4.0768776, -26.8226929, 26.8383942
1: -10.2207327, 6.6966715, -10.1946402, 6.6753254, -13.6129837, 13.6172848
2: -14.3250818, 4.4942408, -14.3206091, 4.4770365, -14.3123703, 14.3288002
3: -21.0081539, 0.8460035, -21.0029602, 0.7969415, -19.5890808, 19.6374168
4: -22.1545448, 3.4067154, -22.1516628, 3.3694382, -19.5872688, 19.6417580
5: -20.5434055, 5.8086338, -20.5420818, 5.7707896, -23.0752945, 23.1395569
6: -22.5456104, 3.2443361, -22.5167065, 3.2150297, -21.2604218, 21.2772446
7: -21.4548626, 4.0425320, -21.4382439, 4.0027323, -21.1423950, 21.1863251
8: -34.1628723, -4.0536919, -34.1517448, -4.0785675, -20.8937225, 20.9027901
9: -12.3238010, 16.6914291, -12.3063402, 16.6493092, -26.4024811, 26.4786301
10: -6.4340200, 20.7235413, -6.4137974, 20.6865196, -23.7222519, 23.7779160
11: -7.0219450, 13.9955063, -6.9555912, 13.9740028, -18.6195564, 18.5594940
12: 0.6025124, 35.3198433, 0.7311406, 35.3238907, -28.7992325, 28.6359329
13: -10.8501091, 24.3130341, -10.7217302, 24.2997093, -30.2252502, 30.1067734
14: -33.2890091, 10.5094700, -33.1051102, 10.4991245, -38.1832123, 37.9747620
15: -20.7442093, 0.3683007, -20.7316189, 0.2992663, -18.6301270, 18.6725807
16: -14.5659256, 7.5452843, -14.5407867, 7.4657807, -22.0317059, 22.0860710
17: -21.4975891, 18.8175316, -21.3237724, 18.8148899, -36.9350891, 36.7461014
18: -14.7322636, 9.5393219, -14.7190218, 9.5099144, -21.1611252, 21.1558189
19: -10.8807373, 6.8305597, -10.8530521, 6.8087678, -14.9808273, 14.9714661
20: -15.1055555, 4.9907327, -15.0807323, 4.9537306, -17.9637337, 17.9801559
21: -11.4142113, 9.6195707, -11.3813381, 9.6000557, -18.6582718, 18.6566391
22: -9.6702938, 7.9033670, -9.6215687, 7.8595805, -15.4363174, 15.4354343
23: -14.0859680, 7.3920078, -14.0639763, 7.3536148, -19.6254387, 19.5950394
24: -17.4927139, 6.2319083, -17.4793434, 6.1889215, -18.7470932, 18.7430077
25: -11.4249363, 10.1729975, -11.3743887, 10.1318817, -20.5272064, 20.5008469
26: -16.2806396, 9.8302088, -16.2162666, 9.8090658, -24.8322296, 24.7485657
27: -27.5506096, 0.8761058, -27.5489311, 0.8077393, -20.3285065, 20.3938179
28: -16.3612785, 7.3427453, -16.3320618, 7.3073096, -20.8663940, 20.8376160
29: -7.4460363, 10.4148922, -7.3798132, 10.3961906, -16.2375336, 16.1859055
30: -19.2005806, 7.4331408, -19.1781807, 7.3887396, -21.8182144, 21.8387070
31: -13.1452074, 9.5381660, -13.1187458, 9.5032225, -19.0601997, 19.0671921
32: -12.5462456, 9.3409176, -12.5155163, 9.3108759, -18.1621246, 18.1853371
33: -45.4993134, -9.2817383, -45.4576225, -9.3920107, -31.3971634, 31.4511223
34: -42.0034981, -13.9880095, -42.0000420, -14.0863113, -19.8735390, 19.9659042
35: -29.0777817, -2.4272213, -29.0532131, -2.4607124, -21.8957672, 21.8436012
36: -23.7693443, 3.8114514, -23.7354202, 3.7717443, -23.6577454, 23.5967216
37: -43.7009048, -4.5776658, -43.6563377, -4.6339641, -36.2554932, 36.1927338
38: -30.0504494, 1.4025931, -30.0130520, 1.3867478, -29.3863983, 29.2763367
39: -38.9786835, -3.8898249, -38.9349899, -3.9676909, -32.5105896, 32.4997559
40: -44.3816414, -12.3176365, -44.3515587, -12.4122505, -26.2800446, 26.3497086
41: -24.2966461, 5.1101336, -24.2735481, 5.0308986, -23.3616028, 23.4232330
42: -19.4886360, 2.2930198, -19.4580288, 2.2683249, -16.6452751, 16.6305752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1706

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4743195
time: 50.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5208538, upper bound: 13.4768668
time: 35.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.1150093, 4.1051702, -29.1127605, 4.0901227, -26.8408508, 26.8548203
1: -10.2216024, 6.6968269, -10.2015419, 6.6866322, -13.6290359, 13.6231499
2: -14.3256865, 4.4940777, -14.3335600, 4.4839993, -14.3216324, 14.3437805
3: -21.0088673, 0.8463430, -21.0299492, 0.8100524, -19.6042747, 19.6753616
4: -22.1553383, 3.4067454, -22.1646137, 3.3752575, -19.6214981, 19.6476631
5: -20.5440388, 5.8104219, -20.5851097, 5.7912011, -23.0995178, 23.1935959
6: -22.5495415, 3.2447376, -22.5377464, 3.2641125, -21.3110580, 21.2942352
7: -21.4559784, 4.0446563, -21.4748039, 4.0153084, -21.1557083, 21.2266083
8: -34.1634674, -4.0546389, -34.1684532, -4.0764623, -20.9036713, 20.9159813
9: -12.3260069, 16.6921043, -12.3253365, 16.6929989, -26.4528961, 26.5003433
10: -6.4346452, 20.7247124, -6.4190216, 20.7324295, -23.7614822, 23.7885590
11: -7.0230932, 14.0023518, -7.0031090, 13.9953651, -18.6380615, 18.6201439
12: 0.5885363, 35.3204422, 0.6861596, 35.4361649, -28.9272385, 28.6736755
13: -10.8590755, 24.3134155, -10.7524490, 24.3629284, -30.2961731, 30.1322327
14: -33.2902985, 10.5108452, -33.1669312, 10.5134802, -38.1974792, 38.0342026
15: -20.7446594, 0.3724995, -20.7550316, 0.3156993, -18.6410294, 18.6987457
16: -14.5683012, 7.5456858, -14.5562162, 7.4874907, -22.0557919, 22.1019020
17: -21.4988651, 18.8181782, -21.3369427, 18.8440628, -36.9724884, 36.7555389
18: -14.7324371, 9.5446720, -14.7655725, 9.5305061, -21.1790161, 21.1999626
19: -10.8813477, 6.8356118, -10.8934669, 6.8225598, -14.9924927, 15.0203629
20: -15.1062889, 4.9982038, -15.1367140, 4.9777160, -17.9850616, 18.0459328
21: -11.4148636, 9.6249294, -11.4320717, 9.6183624, -18.6736832, 18.7152596
22: -9.6705132, 7.9091196, -9.6612968, 7.8800573, -15.4534664, 15.4825611
23: -14.0866032, 7.3986907, -14.1187696, 7.3758183, -19.6444206, 19.6584663
24: -17.4928818, 6.2419171, -17.5461788, 6.2202740, -18.7712517, 18.8207245
25: -11.4253159, 10.1808681, -11.4349413, 10.1619110, -20.5531502, 20.5732880
26: -16.2809143, 9.8357964, -16.2613964, 9.8277979, -24.8520050, 24.7787628
27: -27.5509605, 0.8870144, -27.6325760, 0.8396759, -20.3518486, 20.4900894
28: -16.3622837, 7.3491545, -16.3954277, 7.3271151, -20.8835373, 20.9132652
29: -7.4463167, 10.4189882, -7.4223194, 10.4110432, -16.2516098, 16.2375603
30: -19.2011833, 7.4436359, -19.2472229, 7.4224377, -21.8449020, 21.9242706
31: -13.1457949, 9.5457649, -13.1716595, 9.5260391, -19.0799980, 19.1322708
32: -12.5543814, 9.3410645, -12.5473843, 9.3722153, -18.2322769, 18.2094193
33: -45.5033264, -9.2803040, -45.4833755, -9.3306551, -31.4737473, 31.4773102
34: -42.0037766, -13.9873743, -42.0097961, -14.0804501, -19.8836365, 19.9790878
35: -29.0780258, -2.4265578, -29.0661278, -2.4502952, -21.9220123, 21.8541260
36: -23.7693596, 3.8116837, -23.7498779, 3.7837749, -23.6722336, 23.6128616
37: -43.7071953, -4.5766621, -43.6922684, -4.5661354, -36.3413086, 36.2278137
38: -30.0503845, 1.4032602, -30.0218964, 1.4067659, -29.4098053, 29.2936707
39: -38.9845772, -3.8887329, -38.9641304, -3.8986161, -32.5979919, 32.5282974
40: -44.3888016, -12.3173294, -44.3847351, -12.3552370, -26.3476486, 26.3822174
41: -24.3024292, 5.1105714, -24.3048592, 5.0949841, -23.4380875, 23.4512711
42: -19.4959087, 2.2934823, -19.4819908, 2.3248491, -16.7103577, 16.6490421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1706

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4896064, upper bound: 13.5183004
time: 41.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5208538, upper bound: 13.5208537
time: 35.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 78.92 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4460581
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.5208538, upper bound: 13.4487189
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4900419
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4927048
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.4896064, upper bound: 13.4743195
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.5208538, upper bound: 13.4768668
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.4896064, upper bound: 13.5183004
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 78.92
Output dim: 12, lower bound: -13.5208538, upper bound: 13.5208537

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.1275692, 4.0632319, -29.0923119, 4.0695472, -26.8304901, 26.7811050
1: -10.2025785, 6.6680613, -10.1938095, 6.6714458, -13.5809746, 13.5787029
2: -14.3477812, 4.4654026, -14.3198023, 4.4712734, -14.3294792, 14.2912903
3: -21.0080833, 0.7922585, -20.9979134, 0.7930105, -19.5877686, 19.5744934
4: -22.1514721, 3.3624525, -22.1507721, 3.3650365, -19.5507889, 19.5831375
5: -20.5765190, 5.7659311, -20.5412064, 5.7646837, -23.1048126, 23.0720291
6: -22.5211353, 3.2105465, -22.5158539, 3.2120094, -21.2290573, 21.2266960
7: -21.4730225, 3.9922900, -21.4366417, 3.9950931, -21.1539383, 21.1104813
8: -34.1693420, -4.0919094, -34.1502380, -4.0852633, -20.8818741, 20.8560104
9: -12.3044319, 16.6499863, -12.3019867, 16.6447487, -26.3852234, 26.3991776
10: -6.4094858, 20.6928978, -6.4088788, 20.6807995, -23.7018127, 23.7165413
11: -6.9668474, 13.9795589, -6.9515705, 13.9707022, -18.5509300, 18.5394363
12: 0.7254872, 35.3424759, 0.7410059, 35.3138809, -28.6310272, 28.6529236
13: -10.7261581, 24.2774563, -10.7152596, 24.2917633, -30.0855789, 30.0644913
14: -33.1305847, 10.3902521, -33.0969620, 10.4566536, -37.9681091, 37.8626328
15: -20.7250042, 0.3100266, -20.7266674, 0.2968163, -18.5895462, 18.6161995
16: -14.5494499, 7.4613404, -14.5386810, 7.4609509, -22.0104008, 22.0000210
17: -21.3146553, 18.7161465, -21.3170242, 18.7802868, -36.7114868, 36.6461716
18: -14.6978607, 9.5117130, -14.7109394, 9.5050325, -21.1117249, 21.1266785
19: -10.8661833, 6.8123293, -10.8490658, 6.8060846, -14.9615402, 14.9462509
20: -15.1056271, 4.9587154, -15.0771093, 4.9494648, -17.9596710, 17.9370079
21: -11.3775196, 9.6063194, -11.3755722, 9.5976353, -18.6289902, 18.6147003
22: -9.6109371, 7.8772082, -9.6146212, 7.8584909, -15.3755913, 15.4066353
23: -14.0849924, 7.3602014, -14.0605679, 7.3502750, -19.5884171, 19.5673904
24: -17.4830208, 6.2020884, -17.4763165, 6.1858153, -18.7143402, 18.7175446
25: -11.3713989, 10.1448822, -11.3710413, 10.1307716, -20.4608459, 20.4702644
26: -16.2060509, 9.8312645, -16.2091789, 9.8078289, -24.7239456, 24.7560425
27: -27.5443020, 0.8204594, -27.5446167, 0.8036571, -20.3061905, 20.3206100
28: -16.3520870, 7.3112473, -16.3281384, 7.3031750, -20.8297348, 20.8080063
29: -7.3718081, 10.3852520, -7.3743219, 10.3887520, -16.1562386, 16.1517258
30: -19.1774273, 7.3994770, -19.1741886, 7.3842859, -21.7932816, 21.7975502
31: -13.1305494, 9.5116043, -13.1136322, 9.5001488, -19.0394669, 19.0307503
32: -12.5201464, 9.3095245, -12.5141926, 9.3085232, -18.1243439, 18.1254654
33: -45.4234428, -9.3629732, -45.4391632, -9.3944111, -31.3135452, 31.3516235
34: -41.9607048, -14.0669966, -41.9873581, -14.0879383, -19.8265762, 19.8745880
35: -29.0381165, -2.4388986, -29.0460835, -2.4606433, -21.8189812, 21.8526306
36: -23.7254143, 3.7765527, -23.7317009, 3.7700925, -23.5724640, 23.5838776
37: -43.6263275, -4.6004577, -43.6393509, -4.6360626, -36.1647186, 36.1969376
38: -30.0007820, 1.3820024, -30.0079823, 1.3832793, -29.2817764, 29.2807846
39: -38.9168701, -3.9336736, -38.9224472, -3.9690337, -32.4302368, 32.4642105
40: -44.3057022, -12.4071655, -44.3291245, -12.4143057, -26.2092819, 26.2363167
41: -24.2504578, 5.0385189, -24.2591152, 5.0293856, -23.3248367, 23.3393631
42: -19.4634113, 2.2738571, -19.4553375, 2.2657890, -16.6004486, 16.6030388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1589

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4191768
time: 27.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5204021, upper bound: 13.4482681
time: 43.16 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -29.1504841, 4.1045656, -29.0939426, 4.0743465, -26.8580322, 26.8311005
1: -10.2281799, 6.6980696, -10.1941957, 6.6739187, -13.6096611, 13.6200523
2: -14.3551674, 4.4910331, -14.3198957, 4.4743929, -14.3412285, 14.3212967
3: -21.0254993, 0.8447754, -21.0025005, 0.7955561, -19.6069183, 19.6337090
4: -22.1607285, 3.4064627, -22.1511745, 3.3676291, -19.5719604, 19.6511459
5: -20.5832748, 5.8093600, -20.5411530, 5.7684383, -23.1151886, 23.1358795
6: -22.5468483, 3.2419968, -22.5152054, 3.2130013, -21.2594986, 21.2732849
7: -21.4948864, 4.0385923, -21.4369736, 3.9994144, -21.1825867, 21.1769562
8: -34.1834641, -4.0580120, -34.1514053, -4.0814061, -20.9072189, 20.8934746
9: -12.3227959, 16.7025814, -12.3042202, 16.6486683, -26.3995132, 26.4889297
10: -6.4334126, 20.7401237, -6.4118633, 20.6847115, -23.7170715, 23.7887001
11: -7.0416889, 13.9940624, -6.9544754, 13.9722710, -18.6397133, 18.5558586
12: 0.6074758, 35.3768692, 0.7361264, 35.3233147, -28.7854843, 28.6921997
13: -10.8452425, 24.3153877, -10.7178106, 24.2984505, -30.2178726, 30.1041527
14: -33.3329086, 10.5052242, -33.1040154, 10.4956779, -38.2233276, 37.9676666
15: -20.7440090, 0.3766446, -20.7288380, 0.2981641, -18.6240082, 18.6855545
16: -14.5735559, 7.5480118, -14.5398197, 7.4636397, -22.0371952, 22.0878315
17: -21.5019169, 18.8162708, -21.3229370, 18.8128128, -36.9368744, 36.7439346
18: -14.7330189, 9.5374050, -14.7180195, 9.5075502, -21.1576080, 21.1527786
19: -10.9016638, 6.8286719, -10.8521233, 6.8073521, -15.0026970, 14.9672279
20: -15.1391640, 4.9874916, -15.0797081, 4.9515777, -17.9967728, 17.9747925
21: -11.4233665, 9.6184845, -11.3803787, 9.5989571, -18.6762924, 18.6463203
22: -9.6670208, 7.9120040, -9.6172066, 7.8583803, -15.4321289, 15.4492455
23: -14.1149750, 7.3928862, -14.0631809, 7.3525195, -19.6531906, 19.5916138
24: -17.5059624, 6.2297378, -17.4791679, 6.1868563, -18.7602501, 18.7395706
25: -11.4300613, 10.1749735, -11.3740635, 10.1305513, -20.5321121, 20.5019989
26: -16.2841396, 9.8439178, -16.2139778, 9.8082924, -24.8282547, 24.7653275
27: -27.5627441, 0.8735204, -27.5474472, 0.8054657, -20.3329315, 20.3864250
28: -16.3880119, 7.3397884, -16.3307648, 7.3050933, -20.8934021, 20.8317337
29: -7.4448571, 10.4156656, -7.3766193, 10.3956833, -16.2365341, 16.1806984
30: -19.2095242, 7.4321051, -19.1777763, 7.3868904, -21.8297882, 21.8344688
31: -13.1686382, 9.5362711, -13.1180553, 9.5012913, -19.0837631, 19.0636711
32: -12.5443573, 9.3456745, -12.5126419, 9.3107548, -18.1591263, 18.1859360
33: -45.5003166, -9.2498178, -45.4544220, -9.3935413, -31.3934479, 31.4811096
34: -41.9998169, -13.9657307, -41.9974899, -14.0869055, -19.8634453, 19.9895706
35: -29.0741463, -2.4061308, -29.0495548, -2.4613407, -21.8871155, 21.8686867
36: -23.7658577, 3.8200717, -23.7321339, 3.7715650, -23.6524353, 23.6011314
37: -43.7005081, -4.5414615, -43.6535034, -4.6345463, -36.2512283, 36.2268677
38: -30.0487061, 1.3996496, -30.0111217, 1.3842776, -29.3779678, 29.2728462
39: -38.9765778, -3.8548756, -38.9316940, -3.9690387, -32.5048218, 32.5312500
40: -44.3826904, -12.3077545, -44.3497200, -12.4125738, -26.2758331, 26.3608932
41: -24.2937489, 5.1212406, -24.2706566, 5.0305948, -23.3577576, 23.4318237
42: -19.4890480, 2.3029099, -19.4558277, 2.2676244, -16.6449432, 16.6417179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1589

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4473224
time: 30.60 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5204021, upper bound: 13.4764150
time: 32.26 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -29.1518459, 4.1051192, -29.1112938, 4.0875459, -26.8762360, 26.8475037
1: -10.2290440, 6.6982360, -10.2011089, 6.6852293, -13.6257133, 13.6259308
2: -14.3557816, 4.4909043, -14.3328295, 4.4813557, -14.3505135, 14.3362923
3: -21.0262012, 0.8451128, -21.0294437, 0.8086877, -19.6221046, 19.6716843
4: -22.1615124, 3.4065413, -22.1641235, 3.3734140, -19.6061859, 19.6570473
5: -20.5839176, 5.8111806, -20.5842094, 5.7888622, -23.1394043, 23.1898804
6: -22.5508804, 3.2423897, -22.5362320, 3.2620687, -21.3100967, 21.2902565
7: -21.4960060, 4.0407109, -21.4735661, 4.0120182, -21.1958847, 21.2172623
8: -34.1840553, -4.0589890, -34.1681099, -4.0793157, -20.9171448, 20.9066772
9: -12.3249960, 16.7032509, -12.3232269, 16.6923370, -26.4499664, 26.5106201
10: -6.4340277, 20.7413120, -6.4170961, 20.7306213, -23.7563171, 23.7993431
11: -7.0428267, 14.0009270, -7.0019674, 13.9936724, -18.6582146, 18.6165123
12: 0.5935183, 35.3775215, 0.6911616, 35.4355774, -28.9135590, 28.7299500
13: -10.8541985, 24.3157444, -10.7485151, 24.3616180, -30.2888031, 30.1295624
14: -33.3341827, 10.5065413, -33.1658592, 10.5100708, -38.2376556, 38.0270767
15: -20.7444649, 0.3808687, -20.7522602, 0.3145893, -18.6349182, 18.7117500
16: -14.5759420, 7.5484142, -14.5552692, 7.4853597, -22.0613022, 22.1036835
17: -21.5031738, 18.8169632, -21.3360901, 18.8419418, -36.9742737, 36.7533646
18: -14.7331715, 9.5427380, -14.7645721, 9.5281887, -21.1754799, 21.1969223
19: -10.9022856, 6.8336983, -10.8925705, 6.8211327, -15.0143509, 15.0161324
20: -15.1399031, 4.9949694, -15.1357327, 4.9755797, -18.0181046, 18.0405693
21: -11.4240284, 9.6238174, -11.4311390, 9.6172628, -18.6917267, 18.7049484
22: -9.6672134, 7.9177341, -9.6569128, 7.8788509, -15.4492779, 15.4963875
23: -14.1156359, 7.3995590, -14.1179695, 7.3747492, -19.6721802, 19.6550293
24: -17.5060787, 6.2397242, -17.5460110, 6.2182369, -18.7844200, 18.8172989
25: -11.4304323, 10.1828594, -11.4346333, 10.1605692, -20.5580215, 20.5744057
26: -16.2843742, 9.8494968, -16.2591171, 9.8270140, -24.8480186, 24.7955017
27: -27.5631504, 0.8844242, -27.6310635, 0.8373742, -20.3562851, 20.4826660
28: -16.3890171, 7.3462057, -16.3941116, 7.3248925, -20.9105148, 20.9073944
29: -7.4451103, 10.4197302, -7.4191380, 10.4105453, -16.2505951, 16.2323761
30: -19.2101841, 7.4425650, -19.2467918, 7.4205647, -21.8564606, 21.9200172
31: -13.1692619, 9.5438786, -13.1709900, 9.5240736, -19.1035652, 19.1287613
32: -12.5524940, 9.3458519, -12.5445595, 9.3721275, -18.2292938, 18.2100372
33: -45.5043030, -9.2484093, -45.4801636, -9.3321342, -31.4700394, 31.5072861
34: -42.0000305, -13.9650383, -42.0072289, -14.0810108, -19.8735313, 20.0027542
35: -29.0743637, -2.4054139, -29.0624790, -2.4509225, -21.9133911, 21.8791885
36: -23.7659111, 3.8203268, -23.7465839, 3.7835653, -23.6669235, 23.6172714
37: -43.7068176, -4.5404091, -43.6894379, -4.5666852, -36.3370895, 36.2619858
38: -30.0486469, 1.4002953, -30.0199871, 1.4042668, -29.4013367, 29.2901878
39: -38.9824295, -3.8537626, -38.9608345, -3.8999527, -32.5922470, 32.5598221
40: -44.3898239, -12.3074932, -44.3828773, -12.3555441, -26.3434219, 26.3933563
41: -24.2995071, 5.1216664, -24.3020515, 5.0947056, -23.4342117, 23.4598618
42: -19.4963512, 2.3033752, -19.4798012, 2.3241363, -16.7100334, 16.6601830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1589

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4913095
time: 33.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5204021, upper bound: 13.5204019
time: 28.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 63.93 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4191768
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5204021, upper bound: 13.4482681
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4473224
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5204021, upper bound: 13.4764150
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5185298, upper bound: 13.4913095
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.93
Output dim: 12, lower bound: -13.5204021, upper bound: 13.5204019

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.1271648, 4.0608511, -29.1540527, 4.0693631, -26.8083115, 26.8432999
1: -10.2024632, 6.6666384, -10.2146854, 6.6732702, -13.5734138, 13.5992470
2: -14.3476744, 4.4640093, -14.3686028, 4.4712763, -14.3116798, 14.3385010
3: -21.0078411, 0.7915847, -21.0076122, 0.7941575, -19.5864410, 19.5795403
4: -22.1513481, 3.3593812, -22.1835251, 3.3646173, -19.5369110, 19.6007805
5: -20.5762138, 5.7641554, -20.5497475, 5.7654352, -23.1040115, 23.0692978
6: -22.5179443, 3.2104120, -22.5123787, 3.2532177, -21.2674026, 21.2018547
7: -21.4726334, 3.9914637, -21.4599228, 3.9965971, -21.1434174, 21.1142845
8: -34.1692085, -4.0939975, -34.1909866, -4.0854650, -20.8593407, 20.8964958
9: -12.3040190, 16.6486282, -12.3210564, 16.6473083, -26.3851624, 26.4164581
10: -6.4074712, 20.6917515, -6.4075346, 20.6867218, -23.7019043, 23.7179832
11: -6.9647398, 13.9793959, -6.9537268, 13.9778852, -18.5567818, 18.5412560
12: 0.7268305, 35.3420181, 0.7404733, 35.3620911, -28.6555634, 28.6259918
13: -10.7256298, 24.2757988, -10.7394590, 24.2930946, -30.0846939, 30.0886040
14: -33.1298332, 10.3886833, -33.1273308, 10.4605799, -37.9640045, 37.8985443
15: -20.7247314, 0.3080070, -20.7436104, 0.2994893, -18.5819931, 18.6327629
16: -14.5490150, 7.4604425, -14.5552483, 7.4669490, -22.0159645, 22.0156898
17: -21.3141403, 18.7140465, -21.3236389, 18.7786846, -36.6989594, 36.6566696
18: -14.6970396, 9.5103149, -14.7117109, 9.5087700, -21.1227875, 21.1149025
19: -10.8655491, 6.8114953, -10.8656168, 6.8052750, -14.9570084, 14.9616089
20: -15.1042919, 4.9584975, -15.0803165, 4.9517655, -17.9631386, 17.9395218
21: -11.3757439, 9.6061649, -11.3780556, 9.6140909, -18.6469574, 18.6146698
22: -9.6096001, 7.8767495, -9.6167908, 7.8779011, -15.3906937, 15.4060364
23: -14.0845165, 7.3594408, -14.0788898, 7.3510885, -19.5866394, 19.5883827
24: -17.4826107, 6.2012186, -17.4878273, 6.1914344, -18.7139778, 18.7251053
25: -11.3696823, 10.1443853, -11.3735790, 10.1464510, -20.4660759, 20.4745750
26: -16.2051411, 9.8301220, -16.2216682, 9.8142233, -24.7245331, 24.7593842
27: -27.5429325, 0.8192906, -27.5442791, 0.8113046, -20.3161392, 20.3071785
28: -16.3511543, 7.3100142, -16.3374844, 7.3028989, -20.8285980, 20.8246727
29: -7.3698101, 10.3850670, -7.3759408, 10.4068241, -16.1627159, 16.1505737
30: -19.1743145, 7.3991499, -19.1727390, 7.4351807, -21.8453064, 21.7811127
31: -13.1299362, 9.5095787, -13.1198635, 9.4997473, -19.0401993, 19.0366096
32: -12.5182934, 9.3092813, -12.5180788, 9.3300209, -18.1450462, 18.1159286
33: -45.4203148, -9.3639050, -45.4484406, -9.3843298, -31.3148880, 31.3561935
34: -41.9594841, -14.0672855, -41.9887466, -14.0519419, -19.8588562, 19.8584061
35: -29.0365944, -2.4392245, -29.0538673, -2.4520094, -21.8205948, 21.8571587
36: -23.7240219, 3.7763629, -23.7436237, 3.7720075, -23.5725365, 23.5936508
37: -43.6244774, -4.6012268, -43.6472778, -4.6301775, -36.1683960, 36.2042007
38: -30.0001049, 1.3800180, -30.0283413, 1.3808217, -29.2804337, 29.2876740
39: -38.9165115, -3.9370482, -38.9448090, -3.9728365, -32.4227524, 32.4824295
40: -44.3029404, -12.4076672, -44.3340645, -12.4092007, -26.2112732, 26.2383881
41: -24.2480774, 5.0382481, -24.2664700, 5.0337763, -23.3276367, 23.3410530
42: -19.4592171, 2.2735224, -19.4515572, 2.2717686, -16.5943031, 16.6054592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 685

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5193269, upper bound: 13.4138164
time: 29.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5196711, upper bound: 13.4475378
time: 27.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.1500568, 4.1022549, -29.1557140, 4.0740805, -26.8358383, 26.8932343
1: -10.2280407, 6.6966457, -10.2150688, 6.6757445, -13.6020813, 13.6406021
2: -14.3550415, 4.4896488, -14.3686647, 4.4744110, -14.3234177, 14.3685036
3: -21.0252953, 0.8441088, -21.0121803, 0.7967350, -19.6055984, 19.6387558
4: -22.1605797, 3.4034076, -22.1839561, 3.3671689, -19.5580826, 19.6688156
5: -20.5829849, 5.8076353, -20.5496063, 5.7691655, -23.1143799, 23.1331482
6: -22.5436993, 3.2418852, -22.5117054, 3.2542152, -21.2977600, 21.2484093
7: -21.4944782, 4.0377574, -21.4602242, 4.0009289, -21.1721115, 21.1807632
8: -34.1833000, -4.0601130, -34.1921730, -4.0816441, -20.8846817, 20.9339561
9: -12.3223991, 16.7012272, -12.3232746, 16.6511955, -26.3994751, 26.5062103
10: -6.4313855, 20.7389679, -6.4105363, 20.6905861, -23.7171402, 23.7901726
11: -7.0395231, 13.9939222, -6.9566288, 13.9794617, -18.6455650, 18.5576935
12: 0.6088920, 35.3764534, 0.7356358, 35.3715553, -28.8100739, 28.6652908
13: -10.8447475, 24.3137627, -10.7419701, 24.2997074, -30.2170258, 30.1282539
14: -33.3322067, 10.5035896, -33.1343956, 10.4996281, -38.2192841, 38.0035934
15: -20.7437172, 0.3746414, -20.7457619, 0.3008361, -18.6164780, 18.7021179
16: -14.5731211, 7.5471134, -14.5563812, 7.4696326, -22.0427532, 22.1034946
17: -21.5014381, 18.8141575, -21.3295784, 18.8111610, -36.9244232, 36.7544556
18: -14.7322311, 9.5359564, -14.7187767, 9.5112877, -21.1686325, 21.1410370
19: -10.9010248, 6.8278475, -10.8686819, 6.8065310, -14.9981766, 14.9825630
20: -15.1378288, 4.9873104, -15.0829287, 4.9538765, -18.0002136, 17.9773216
21: -11.4215622, 9.6183224, -11.3828726, 9.6154356, -18.6942978, 18.6462860
22: -9.6656914, 7.9115400, -9.6193504, 7.8777943, -15.4472179, 15.4486275
23: -14.1145325, 7.3921280, -14.0814905, 7.3533144, -19.6514130, 19.6125603
24: -17.5055580, 6.2288857, -17.4906559, 6.1924429, -18.7598991, 18.7470894
25: -11.4283342, 10.1744871, -11.3766289, 10.1462326, -20.5373611, 20.5062943
26: -16.2831955, 9.8427467, -16.2264748, 9.8147058, -24.8288803, 24.7686310
27: -27.5614071, 0.8723059, -27.5471268, 0.8131361, -20.3428917, 20.3729782
28: -16.3870735, 7.3385563, -16.3400803, 7.3048310, -20.8922577, 20.8483543
29: -7.4428511, 10.4154625, -7.3782430, 10.4137592, -16.2430153, 16.1795464
30: -19.2064075, 7.4317703, -19.1763191, 7.4378109, -21.8817635, 21.8179855
31: -13.1680593, 9.5342522, -13.1242981, 9.5008717, -19.0845146, 19.0695496
32: -12.5424881, 9.3454294, -12.5165615, 9.3322811, -18.1798363, 18.1764183
33: -45.4972458, -9.2507534, -45.4636688, -9.3834505, -31.3947906, 31.4856644
34: -41.9985962, -13.9659719, -41.9989014, -14.0509081, -19.8957596, 19.9734192
35: -29.0726433, -2.4064319, -29.0573292, -2.4527013, -21.8887863, 21.8731918
36: -23.7644920, 3.8198593, -23.7440262, 3.7735109, -23.6525269, 23.6108742
37: -43.6987000, -4.5422177, -43.6615067, -4.6286492, -36.2549515, 36.2340698
38: -30.0480270, 1.3976712, -30.0315094, 1.3818583, -29.3766403, 29.2796936
39: -38.9761658, -3.8582227, -38.9540863, -3.9728279, -32.4973373, 32.5494003
40: -44.3799553, -12.3082104, -44.3546982, -12.4074736, -26.2778091, 26.3629112
41: -24.2913551, 5.1209364, -24.2779846, 5.0350208, -23.3605576, 23.4335480
42: -19.4848652, 2.3025417, -19.4520817, 2.2735872, -16.6388054, 16.6441269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1691

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4852742, upper bound: 13.4731552
time: 25.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5171390, upper bound: 13.4731552
time: 19.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.1514053, 4.1027613, -29.1730232, 4.0873594, -26.8540573, 26.9096451
1: -10.2289095, 6.6967964, -10.2219744, 6.6870527, -13.6181297, 13.6464806
2: -14.3556576, 4.4895449, -14.3816490, 4.4813843, -14.3326836, 14.3835106
3: -21.0259838, 0.8444471, -21.0391369, 0.8098574, -19.6207695, 19.6767159
4: -22.1613712, 3.4034858, -22.1968555, 3.3729906, -19.5923080, 19.6747055
5: -20.5836163, 5.8094430, -20.5926857, 5.7896194, -23.1385956, 23.1871719
6: -22.5476456, 3.2422781, -22.5327625, 3.3032856, -21.3484344, 21.2653961
7: -21.4955788, 4.0398755, -21.4968300, 4.0135417, -21.1854324, 21.2210846
8: -34.1839027, -4.0611176, -34.2088852, -4.0795321, -20.8946457, 20.9471588
9: -12.3246193, 16.7018547, -12.3422918, 16.6949272, -26.4499130, 26.5278854
10: -6.4320297, 20.7401390, -6.4157681, 20.7365341, -23.7563324, 23.8007965
11: -7.0407019, 14.0007610, -7.0041428, 14.0008888, -18.6640854, 18.6183586
12: 0.5948725, 35.3770370, 0.6906075, 35.4838219, -28.9381104, 28.7030334
13: -10.8536940, 24.3141174, -10.7727108, 24.3629150, -30.2879028, 30.1536713
14: -33.3334122, 10.5049648, -33.1962547, 10.5139837, -38.2335968, 38.0630264
15: -20.7441788, 0.3788683, -20.7691956, 0.3172925, -18.6273804, 18.7283401
16: -14.5754986, 7.5475149, -14.5717993, 7.4913340, -22.0668335, 22.1193142
17: -21.5026646, 18.8148346, -21.3427277, 18.8403702, -36.9618073, 36.7639084
18: -14.7323923, 9.5413065, -14.7653542, 9.5318966, -21.1865120, 21.1851654
19: -10.9016399, 6.8328505, -10.9090986, 6.8203206, -15.0098152, 15.0314789
20: -15.1385403, 4.9947505, -15.1389332, 4.9778948, -18.0215759, 18.0430717
21: -11.4222326, 9.6236858, -11.4336090, 9.6337328, -18.7097168, 18.7049103
22: -9.6658897, 7.9172754, -9.6590786, 7.8982716, -15.4643822, 15.4957752
23: -14.1151571, 7.3987875, -14.1362829, 7.3755865, -19.6704063, 19.6760025
24: -17.5056496, 6.2388740, -17.5575237, 6.2238302, -18.7840271, 18.8248329
25: -11.4287224, 10.1823692, -11.4371910, 10.1762667, -20.5632668, 20.5787315
26: -16.2834740, 9.8483486, -16.2716141, 9.8334351, -24.8486328, 24.7988434
27: -27.5617638, 0.8832197, -27.6307468, 0.8450227, -20.3662415, 20.4692421
28: -16.3880920, 7.3449812, -16.4034348, 7.3246241, -20.9093857, 20.9239922
29: -7.4431114, 10.4195614, -7.4207621, 10.4286203, -16.2570457, 16.2312241
30: -19.2070580, 7.4422703, -19.2453365, 7.4714909, -21.9084740, 21.9035797
31: -13.1686497, 9.5418453, -13.1772308, 9.5236979, -19.1042976, 19.1346588
32: -12.5506401, 9.3456144, -12.5484610, 9.3936424, -18.2499847, 18.2005005
33: -45.5012016, -9.2493458, -45.4894791, -9.3220387, -31.4713669, 31.5118332
34: -41.9988899, -13.9653254, -42.0086861, -14.0450029, -19.9058456, 19.9865761
35: -29.0728416, -2.4057395, -29.0702705, -2.4422901, -21.9150085, 21.8837357
36: -23.7645340, 3.8201218, -23.7584801, 3.7855151, -23.6670151, 23.6270256
37: -43.7049637, -4.5411944, -43.6974297, -4.5608234, -36.3408051, 36.2692413
38: -30.0480042, 1.3983355, -30.0403843, 1.4018679, -29.4000320, 29.2970657
39: -38.9820786, -3.8571186, -38.9832039, -3.9037564, -32.5847931, 32.5780182
40: -44.3871307, -12.3079348, -44.3878708, -12.3504181, -26.3454514, 26.3954430
41: -24.2970734, 5.1213799, -24.3093262, 5.0991492, -23.4370651, 23.4615860
42: -19.4921665, 2.3029943, -19.4760323, 2.3300982, -16.7039185, 16.6625938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1691

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4852742, upper bound: 13.5171389
time: 36.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5171390, upper bound: 13.5171389
time: 29.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 68.93 seconds
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.5193269, upper bound: 13.4138164
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.5196711, upper bound: 13.4475378
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.4852742, upper bound: 13.4731552
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.5171390, upper bound: 13.4731552
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.4852742, upper bound: 13.5171389
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 68.93
Output dim: 12, lower bound: -13.5171390, upper bound: 13.5171389

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.1160126, 4.0514774, -29.1487083, 4.0648122, -26.7924194, 26.8259048
1: -10.1986952, 6.6619258, -10.2128801, 6.6710157, -13.5664864, 13.5903587
2: -14.3430061, 4.4572344, -14.3663530, 4.4680204, -14.3041382, 14.3292961
3: -21.0038452, 0.7801170, -21.0056686, 0.7886574, -19.5653076, 19.5583076
4: -22.1442833, 3.3539290, -22.1801567, 3.3620286, -19.5232811, 19.5825386
5: -20.5685806, 5.7571173, -20.5457058, 5.7619858, -23.0842056, 23.0521088
6: -22.5079956, 3.2083664, -22.5075836, 3.2522392, -21.2564240, 21.1945915
7: -21.4620094, 3.9857101, -21.4544983, 3.9938076, -21.1301651, 21.1030350
8: -34.1628418, -4.1046185, -34.1879425, -4.0905118, -20.8465195, 20.8805008
9: -12.2947874, 16.6305885, -12.3166113, 16.6386757, -26.3671341, 26.3937225
10: -6.4023008, 20.6621475, -6.4050689, 20.6723099, -23.6816940, 23.6954079
11: -6.9372263, 13.9762440, -6.9405127, 13.9763784, -18.5281906, 18.5256157
12: 0.7320533, 35.2943192, 0.7429504, 35.3389816, -28.6260376, 28.5735245
13: -10.7198963, 24.2289085, -10.7366943, 24.2705650, -30.0563889, 30.0377960
14: -33.1188049, 10.3756561, -33.1219940, 10.4539671, -37.9447021, 37.8755569
15: -20.7164249, 0.3038170, -20.7396240, 0.2974861, -18.5687790, 18.6230202
16: -14.5422812, 7.4539709, -14.5520058, 7.4638052, -22.0060863, 22.0059776
17: -21.3084717, 18.6562748, -21.3209057, 18.7509842, -36.6656647, 36.5943985
18: -14.6507092, 9.5048199, -14.6894941, 9.5061054, -21.0762253, 21.0861893
19: -10.8376522, 6.8097577, -10.8522110, 6.8044481, -14.9287605, 14.9462776
20: -15.0643463, 4.9555907, -15.0611305, 4.9503813, -17.9208145, 17.9167671
21: -11.3469677, 9.6039562, -11.3641920, 9.6130342, -18.6153564, 18.5975876
22: -9.5944138, 7.8718362, -9.6094532, 7.8755388, -15.3739128, 15.3941574
23: -14.0617504, 7.3542881, -14.0679598, 7.3486166, -19.5609474, 19.5714607
24: -17.4535084, 6.1973867, -17.4738846, 6.1895175, -18.6830788, 18.7076035
25: -11.3430653, 10.1379948, -11.3608131, 10.1433620, -20.4357643, 20.4549026
26: -16.1830711, 9.8280382, -16.2110806, 9.8132591, -24.6988907, 24.7444382
27: -27.4863491, 0.8164358, -27.5170898, 0.8099642, -20.2561264, 20.2761993
28: -16.3171654, 7.3055220, -16.3211384, 7.3007245, -20.7918930, 20.8032303
29: -7.3490553, 10.3822775, -7.3659143, 10.4054880, -16.1406784, 16.1377068
30: -19.1584320, 7.3915424, -19.1651039, 7.4315319, -21.8249397, 21.7648163
31: -13.0875511, 9.5069923, -13.0995092, 9.4984770, -18.9969330, 19.0138893
32: -12.5094299, 9.2929764, -12.5138168, 9.3221989, -18.1282005, 18.0955467
33: -45.4097252, -9.4157877, -45.4433365, -9.4094086, -31.2795029, 31.2976456
34: -41.9529419, -14.1023216, -41.9856033, -14.0687466, -19.8347816, 19.8203773
35: -29.0293694, -2.4644189, -29.0503845, -2.4641705, -21.8001556, 21.8258286
36: -23.7135544, 3.7734168, -23.7385426, 3.7705631, -23.5586624, 23.5823898
37: -43.6103821, -4.6548562, -43.6405296, -4.6559081, -36.1297760, 36.1437988
38: -29.9752750, 1.3745797, -30.0162735, 1.3782372, -29.2584915, 29.2708969
39: -38.9090347, -3.9774532, -38.9412117, -3.9922388, -32.3964005, 32.4377136
40: -44.2867699, -12.4453764, -44.3263092, -12.4273586, -26.1759567, 26.1916504
41: -24.2381382, 5.0226607, -24.2616081, 5.0262651, -23.3091888, 23.3194542
42: -19.4518719, 2.2602978, -19.4480038, 2.2654219, -16.5809326, 16.5925522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=234, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1691

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5160748, upper bound: 13.3786832
time: 25.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5160748, upper bound: 13.4105500
time: 31.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -29.1381493, 4.0623946, -29.1529236, 4.0655484, -26.8222122, 26.8411179
1: -10.2043152, 6.6771841, -10.2141409, 6.6725836, -13.5791626, 13.6040878
2: -14.3460808, 4.4689007, -14.3662262, 4.4703732, -14.3101654, 14.3395958
3: -21.0117207, 0.7960186, -21.0068130, 0.7926173, -19.5760956, 19.5821304
4: -22.1529350, 3.3637652, -22.1821404, 3.3632932, -19.5476685, 19.5900612
5: -20.5695419, 5.7660084, -20.5446568, 5.7648478, -23.0928268, 23.0667877
6: -22.5123100, 3.2092695, -22.5049267, 3.2530708, -21.2627411, 21.1936378
7: -21.4661789, 4.0047550, -21.4547100, 3.9960041, -21.1370163, 21.1198502
8: -34.1720581, -4.0876193, -34.1898499, -4.0873942, -20.8626862, 20.8986626
9: -12.3114996, 16.6434135, -12.3202267, 16.6403656, -26.3867264, 26.4100876
10: -6.4302731, 20.6870880, -6.4070044, 20.6791649, -23.6758804, 23.7249184
11: -6.9640675, 14.0084305, -6.9512873, 13.9774876, -18.5532494, 18.5685005
12: 0.6592007, 35.3411942, 0.7417207, 35.3604507, -28.7206955, 28.6127396
13: -10.7928658, 24.2754383, -10.7381248, 24.2908726, -30.1504211, 30.0778427
14: -33.1308861, 10.3812571, -33.1259499, 10.4538879, -37.9696350, 37.8893890
15: -20.7296944, 0.3068674, -20.7430382, 0.2958875, -18.5857315, 18.6340294
16: -14.5575895, 7.4740853, -14.5539417, 7.4646759, -22.0222664, 22.0280266
17: -21.3772469, 18.7133141, -21.3221817, 18.7764778, -36.7620697, 36.6450424
18: -14.6978159, 9.5412397, -14.7101851, 9.5077610, -21.1138649, 21.1292725
19: -10.8684292, 6.8427181, -10.8642139, 6.8047419, -14.9522934, 14.9900703
20: -15.1067791, 5.0069151, -15.0785809, 4.9512677, -17.9536362, 17.9856911
21: -11.3775511, 9.6422443, -11.3764248, 9.6136990, -18.6433067, 18.6475182
22: -9.6185474, 7.8860493, -9.6154804, 7.8761840, -15.3961678, 15.4137001
23: -14.0874004, 7.3875794, -14.0778341, 7.3503118, -19.5879669, 19.6142006
24: -17.4834003, 6.2383361, -17.4868393, 6.1905384, -18.7048454, 18.7611427
25: -11.3727837, 10.1778679, -11.3725243, 10.1449547, -20.4605598, 20.5066948
26: -16.2197075, 9.8577881, -16.2202740, 9.8136330, -24.7304077, 24.7830048
27: -27.5393829, 0.8779464, -27.5415821, 0.8108039, -20.2941399, 20.3620529
28: -16.3533516, 7.3502598, -16.3353176, 7.3021102, -20.8215942, 20.8626595
29: -7.3768435, 10.3971958, -7.3740687, 10.4060717, -16.1673203, 16.1607819
30: -19.1750622, 7.4210224, -19.1715813, 7.4346542, -21.8438148, 21.8016090
31: -13.1332932, 9.5594139, -13.1181469, 9.4990721, -19.0330811, 19.0840263
32: -12.5382328, 9.3092651, -12.5163031, 9.3295021, -18.1638718, 18.1072807
33: -45.4868126, -9.3657255, -45.4472847, -9.3873272, -31.3798447, 31.3426208
34: -42.0026093, -14.0670319, -41.9877892, -14.0534134, -19.8978996, 19.8406029
35: -29.0731773, -2.4407327, -29.0529594, -2.4536724, -21.8555832, 21.8432999
36: -23.7379265, 3.7768466, -23.7394886, 3.7715385, -23.5969391, 23.5830688
37: -43.6920013, -4.6041045, -43.6462784, -4.6325798, -36.2394333, 36.1862717
38: -30.0065880, 1.3975525, -30.0226517, 1.3801694, -29.2899017, 29.2837029
39: -38.9777031, -3.9391785, -38.9434853, -3.9750633, -32.4846954, 32.4755554
40: -44.3452415, -12.4081993, -44.3325958, -12.4105177, -26.2523651, 26.2305717
41: -24.2665138, 5.0380192, -24.2650032, 5.0330706, -23.3449173, 23.3344688
42: -19.4761181, 2.2706928, -19.4501648, 2.2687750, -16.5985527, 16.6052780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=234, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1691

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5164102, upper bound: 13.4124057
time: 38.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5164102, upper bound: 13.4442725
time: 29.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 69.24 seconds
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 69.24
Output dim: 12, lower bound: -13.5160748, upper bound: 13.3786832
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 69.24
Output dim: 12, lower bound: -13.5160748, upper bound: 13.4105500
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 69.24
Output dim: 12, lower bound: -13.5164102, upper bound: 13.4124057
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 69.24
Output dim: 12, lower bound: -13.5164102, upper bound: 13.4442725

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 49.57 + 1031.07 = 1080.64 seconds
