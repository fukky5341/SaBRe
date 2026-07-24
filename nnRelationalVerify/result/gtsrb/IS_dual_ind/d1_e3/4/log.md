## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005)
1: (-8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0786438, 23.0786438)
2: (-7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3608322, 20.3608322)
3: (-10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5628357, 22.5628357)
4: (-8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0970383, 22.0970383)
5: (-10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7618561, 24.7618561)
6: (-18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7960205, 18.7960205)
7: (-13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1452637, 26.1452713)
8: (-15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7275085, 26.7275009)
9: (-9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0126572, 21.0126572)
10: (-21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7771988, 28.7771988)
11: (-22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0403061, 25.0403061)
12: (-22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1546097, 26.1546097)
13: (-17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028)
14: (-39.4582977, -7.1018515, -39.4582977, -7.1018515, -27.0013657, 27.0013657)
15: (-11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2471008, 21.2471008)
16: (-20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788)
17: (-40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330)
18: (-16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593)
19: (-15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8187408, 18.8187447)
20: (-13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672)
21: (-19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7662048, 22.7662048)
22: (-15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9356537, 21.9356537)
23: (-15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4218292, 20.4218292)
24: (-11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9382782, 18.9382782)
25: (-13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5915985, 21.5915985)
26: (-23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915)
27: (-14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3851891)
28: (-15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9924622, 21.9924622)
29: (-17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2202301, 21.2202301)
30: (-19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4205017, 21.4205017)
31: (-17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4178162, 22.4178162)
32: (-14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2262497, 19.2262497)
33: (-24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7911987, 32.7911911)
34: (-23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.5005493, 23.5005493)
35: (-19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.6042709, 26.6042633)
36: (-18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2932129, 27.2932205)
37: (-29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4150085, 33.4150085)
38: (-22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9117432, 34.9117432)
39: (-28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1591797, 39.1591949)
40: (-25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8880768, 28.8880768)
41: (-14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6355438, 19.6355438)
42: (-14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0274429, 16.0274429)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 63.30 = 66.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -10.5314132, upper bound: 10.5314132

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5301919, upper bound: 10.5069744
time: 71.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5301989, upper bound: 10.5301987
time: 29.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 101.17 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 101.17
Output dim: 4, lower bound: -10.5301919, upper bound: 10.5069744
IS_A2, status: Status.UNKNOWN, split count: 1, time: 101.17
Output dim: 4, lower bound: -10.5301989, upper bound: 10.5301987

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.8569069, 14.4236851, -14.8593950, 14.4375896, -29.2944965, 29.2830811
1: -8.3256702, 15.3525753, -8.3269615, 15.3603487, -23.4253769, 23.4236755
2: -7.1266918, 13.7590904, -7.1277075, 13.7817783, -19.5570602, 19.5774841
3: -10.5459261, 12.4745693, -10.5476274, 12.4866486, -21.2926102, 21.3285217
4: -8.0815458, 14.8808937, -8.0831928, 14.9042797, -20.8579865, 20.9176102
5: -10.6569176, 14.4658918, -10.6588144, 14.4817305, -23.5373383, 23.5445480
6: -18.7112484, 2.6741223, -18.7401752, 2.6758084, -18.9401169, 18.7759399
7: -13.6157103, 13.8682451, -13.6170568, 13.8780394, -25.6390686, 25.6369934
8: -15.1538544, 12.4618139, -15.1548939, 12.4804678, -26.1920853, 26.1982422
9: -9.8060150, 11.8735466, -9.8137798, 11.8751144, -21.3351593, 21.3200912
10: -21.2200813, 8.8009129, -21.2227001, 8.8117218, -28.7368164, 28.7233582
11: -22.3608017, 4.9582748, -22.3729706, 4.9623775, -24.7163620, 24.8373184
12: -22.4960594, 5.3710961, -22.5126820, 5.3748350, -25.7290421, 25.6430893
13: -17.6435375, 15.5514069, -17.6467705, 15.5652847, -32.6422958, 32.6207733
14: -39.4545059, -7.1554184, -39.4564285, -7.1272316, -26.8184967, 26.8322525
15: -11.9953842, 11.4588823, -11.9974470, 11.4758434, -21.0187225, 21.1305466
16: -20.1255493, 8.1362762, -20.1361847, 8.1394691, -28.2650185, 28.2724609
17: -40.6495361, -1.7928410, -40.6519241, -1.7574902, -38.8713837, 38.8590851
18: -16.6305943, 11.9989958, -16.6333923, 12.0078506, -28.6384449, 28.6323891
19: -15.8833103, 3.8658118, -15.8880062, 3.8672519, -18.8593597, 18.8751678
20: -13.3913441, 4.4979916, -13.3961639, 4.4993920, -17.8907356, 17.8941555
21: -19.0135269, 4.2704391, -19.0252190, 4.2726860, -22.6648941, 22.7388229
22: -15.5997562, 6.8127413, -15.6028147, 6.8174763, -21.9070358, 22.0003357
23: -15.0670013, 5.9878016, -15.0722275, 5.9898376, -19.7932129, 19.8262634
24: -11.5654354, 8.5787373, -11.5674076, 8.5858068, -18.5523911, 18.5894089
25: -13.8912096, 8.4354210, -13.8935080, 8.4390154, -21.3912659, 21.4235611
26: -23.3116951, 7.7565064, -23.3146839, 7.7606492, -31.0723438, 31.0711899
27: -14.6547413, 8.7009754, -14.6677132, 8.7032204, -23.3070526, 23.3219986
28: -15.3623037, 7.8476477, -15.3748360, 7.8490653, -21.1978607, 21.2155609
29: -17.4947624, 4.4225779, -17.5000725, 4.4244871, -21.0832214, 21.0885468
30: -19.3944588, 4.7352180, -19.4097748, 4.7376609, -21.0057678, 21.0347061
31: -17.0556412, 5.7822781, -17.0576897, 5.7881470, -22.8437881, 22.8399677
32: -14.4929495, 6.3072715, -14.5179272, 6.3096070, -19.4419365, 19.4061470
33: -24.2227859, 10.9104557, -24.2297287, 10.9127283, -32.8083038, 32.8034821
34: -23.4743843, 4.3512683, -23.4888878, 4.3521848, -23.3023529, 23.3153229
35: -19.5752468, 9.0812435, -19.5825844, 9.0818548, -26.6140289, 26.5830307
36: -18.4342175, 11.1540661, -18.4515305, 11.1548367, -27.2990875, 27.3162613
37: -29.1109390, 6.9328332, -29.1327553, 6.9343290, -32.8369751, 32.7037506
38: -22.7651997, 14.2621813, -22.7758904, 14.2631159, -34.6235046, 34.6081390
39: -28.0098991, 12.0055408, -28.0144272, 12.0072851, -39.5447388, 39.5427246
40: -25.7728271, 5.3523760, -25.7975349, 5.3539925, -29.0243683, 28.9160995
41: -14.7402210, 7.3645439, -14.7678051, 7.3659053, -19.2688446, 19.1842766
42: -14.6615744, 2.1995697, -14.6862049, 2.2012041, -15.3472023, 15.2482605

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5290747, upper bound: 10.4811839
time: 33.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5290747, upper bound: 10.4811839
time: 36.06 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.8913145, 14.4463787, -14.8610144, 14.4475155, -29.3388290, 29.3073921
1: -8.3407555, 15.3654099, -8.3278923, 15.3659019, -23.0889359, 23.4317169
2: -7.1768289, 13.8002262, -7.1284642, 13.7996750, -19.6674347, 20.3564606
3: -10.5761404, 12.4965639, -10.5491734, 12.4956961, -21.3847733, 22.5524445
4: -8.1381273, 14.9220057, -8.0843544, 14.9226828, -21.0147095, 22.0915909
5: -10.6936550, 14.4962063, -10.6602983, 14.4948759, -23.6101074, 24.7529755
6: -18.7648239, 2.7337241, -18.7626991, 2.6769295, -18.9875641, 18.8507996
7: -13.6353798, 13.8863726, -13.6180859, 13.8856878, -25.6688461, 26.1398392
8: -15.1897182, 12.4939384, -15.1557407, 12.4948301, -26.2673264, 26.7238770
9: -9.8201895, 11.8797674, -9.8146706, 11.8759937, -21.0127716, 21.3528366
10: -21.2328873, 8.8208132, -21.2245159, 8.8186646, -28.7692871, 28.7690964
11: -22.3848000, 4.9812698, -22.3825073, 4.9659281, -25.0390396, 24.7543716
12: -22.5273438, 5.4130530, -22.5266953, 5.3775163, -26.1511383, 25.7990341
13: -17.6712265, 15.5780411, -17.6490650, 15.5762367, -32.6714172, 33.2271042
14: -39.5142212, -7.1049442, -39.4577065, -7.1053181, -26.9395523, 26.9829636
15: -12.0355167, 11.4897223, -11.9988031, 11.4902811, -21.1990204, 21.2416344
16: -20.1471405, 8.1520042, -20.1431580, 8.1417322, -28.2888718, 28.2951622
17: -40.7211418, -1.7322426, -40.6535454, -1.7298126, -38.9913292, 38.9213028
18: -16.6411209, 12.0131559, -16.6348801, 12.0104389, -28.6515598, 28.6480370
19: -15.8926430, 3.8718534, -15.8907518, 3.8683052, -18.8703308, 18.8267632
20: -13.3980684, 4.5100441, -13.3987169, 4.5004711, -17.8985405, 17.9087601
21: -19.0357265, 4.2927251, -19.0341148, 4.2745781, -22.7647858, 22.7088623
22: -15.6154518, 6.8201790, -15.6050520, 6.8200951, -22.0209198, 21.9323120
23: -15.0756645, 5.9968481, -15.0753164, 5.9911652, -20.4138336, 19.8103943
24: -11.5741873, 8.5902710, -11.5687151, 8.5877914, -18.9391632, 18.6039619
25: -13.8948097, 8.4470959, -13.8942356, 8.4413366, -21.5881653, 21.4030762
26: -23.3192558, 7.7591119, -23.3168144, 7.7599945, -31.0792503, 31.0759258
27: -14.6795235, 8.7255001, -14.6767178, 8.7047501, -23.3808899, 23.3715591
28: -15.3852262, 7.8704004, -15.3840084, 7.8498769, -21.9831467, 21.2465363
29: -17.5059891, 4.4280758, -17.5023880, 4.4258261, -21.2154312, 21.0974045
30: -19.4217911, 4.7696199, -19.4215546, 4.7396073, -21.4154129, 21.0675163
31: -17.0607529, 5.7936945, -17.0588722, 5.7904234, -22.8511772, 22.4170609
32: -14.5418482, 6.3542404, -14.5379610, 6.3111730, -19.4863930, 19.2659416
33: -24.2349968, 10.9311342, -24.2342072, 10.9145088, -32.8220520, 32.8052139
34: -23.5013371, 4.3753867, -23.4995193, 4.3527250, -23.4963531, 23.3474731
35: -19.5898476, 9.0907526, -19.5877571, 9.0822229, -26.6022949, 26.6008453
36: -18.4703579, 11.1803026, -18.4657078, 11.1552925, -27.3360672, 27.3159409
37: -29.1500492, 6.9774160, -29.1496468, 6.9354868, -33.4100189, 32.9207993
38: -22.7863922, 14.2648993, -22.7829971, 14.2632713, -34.9095001, 34.6452942
39: -28.0202522, 12.0134325, -28.0175972, 12.0086384, -39.5486450, 39.1601257
40: -25.8157272, 5.3969727, -25.8166122, 5.3549089, -29.0646286, 28.9255447
41: -14.7921600, 7.4208288, -14.7892637, 7.3667059, -19.6282883, 19.3745461
42: -14.7059221, 2.2524445, -14.7055225, 2.2022200, -16.0169678, 15.4471779

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5290815, upper bound: 10.5044222
time: 43.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5290815, upper bound: 10.5291495
time: 35.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 81.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 81.41
Output dim: 4, lower bound: -10.5290747, upper bound: 10.4811839
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 81.41
Output dim: 4, lower bound: -10.5290747, upper bound: 10.4811839
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 81.41
Output dim: 4, lower bound: -10.5290815, upper bound: 10.5044222
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 81.41
Output dim: 4, lower bound: -10.5290815, upper bound: 10.5291495

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.8326159, 14.4224043, -14.7977972, 14.4344158, -29.2670326, 29.2202015
1: -8.3082705, 15.3516064, -8.2828398, 15.3578815, -23.4055328, 23.3785324
2: -7.1121311, 13.7579384, -7.0907688, 13.7789259, -19.5398331, 19.5395432
3: -10.5315819, 12.4731770, -10.5111809, 12.4831181, -21.2740326, 21.2898102
4: -8.0615911, 14.8799877, -8.0326443, 14.9019928, -20.8350143, 20.8651505
5: -10.6408930, 14.4639072, -10.6180773, 14.4767570, -23.5156403, 23.5012741
6: -18.7095470, 2.6648211, -18.7358379, 2.6523576, -18.9110947, 18.7605553
7: -13.5955353, 13.8670721, -13.5666294, 13.8750067, -25.6154938, 25.5847015
8: -15.1345987, 12.4605598, -15.1061907, 12.4772749, -26.1696167, 26.1485748
9: -9.7880936, 11.8722820, -9.7682934, 11.8719311, -21.3126450, 21.2708359
10: -21.2023602, 8.7979584, -21.1779480, 8.8042879, -28.7103500, 28.6717834
11: -22.3566799, 4.9539413, -22.3626633, 4.9514999, -24.7002640, 24.8215866
12: -22.4936810, 5.3608465, -22.5067558, 5.3489814, -25.6997452, 25.6258698
13: -17.6252651, 15.5482063, -17.6003685, 15.5572367, -32.6136780, 32.5667114
14: -39.4378357, -7.1581860, -39.4145584, -7.1341496, -26.7935104, 26.7831497
15: -11.9855051, 11.4576111, -11.9726095, 11.4726267, -21.0048447, 21.1038589
16: -20.1061726, 8.1344519, -20.0871029, 8.1348648, -28.2410374, 28.2215538
17: -40.6295242, -1.7954159, -40.6016006, -1.7639103, -38.8430939, 38.8061829
18: -16.6277161, 11.9877901, -16.6261311, 11.9794674, -28.6071835, 28.6139221
19: -15.8803844, 3.8547792, -15.8806782, 3.8392887, -18.8250275, 18.8565788
20: -13.3890953, 4.4880371, -13.3905411, 4.4742842, -17.8633804, 17.8785782
21: -19.0097046, 4.2609005, -19.0155067, 4.2487097, -22.6277771, 22.7172775
22: -15.5959892, 6.8015409, -15.5933199, 6.7892990, -21.8714752, 21.9785385
23: -15.0645809, 5.9744339, -15.0661898, 5.9559879, -19.7557678, 19.8065033
24: -11.5633116, 8.5659504, -11.5620708, 8.5533762, -18.5137787, 18.5699272
25: -13.8887501, 8.4212484, -13.8872795, 8.4032631, -21.3503799, 21.4020920
26: -23.3080177, 7.7403140, -23.3054390, 7.7196264, -31.0276451, 31.0457535
27: -14.6521931, 8.6859255, -14.6612358, 8.6650896, -23.2557068, 23.2964478
28: -15.3601942, 7.8306789, -15.3694992, 7.8060513, -21.1515656, 21.1927261
29: -17.4910030, 4.4129272, -17.4906139, 4.4000216, -21.0548325, 21.0691757
30: -19.3926926, 4.7271185, -19.4052734, 4.7172394, -20.9796371, 21.0191917
31: -17.0526810, 5.7679667, -17.0502396, 5.7518997, -22.8045807, 22.8182068
32: -14.4901733, 6.2972803, -14.5109615, 6.2844019, -19.4098854, 19.3882294
33: -24.2186852, 10.8965645, -24.2193928, 10.8776035, -32.7689667, 32.7785263
34: -23.4719791, 4.3343477, -23.4829750, 4.3092861, -23.2534485, 23.2901535
35: -19.5724564, 9.0654421, -19.5756073, 9.0416737, -26.5705872, 26.5594559
36: -18.4313049, 11.1361580, -18.4440956, 11.1093788, -27.2499084, 27.2907104
37: -29.1066322, 6.9201355, -29.1219711, 6.9023943, -32.8000031, 32.6793747
38: -22.7613850, 14.2442932, -22.7663307, 14.2176685, -34.5726929, 34.5800552
39: -28.0048161, 11.9978590, -28.0015907, 11.9877901, -39.5161591, 39.5198669
40: -25.7692146, 5.3446798, -25.7884407, 5.3348927, -28.9993744, 28.8982925
41: -14.7368412, 7.3517642, -14.7592793, 7.3334818, -19.2323608, 19.1633072
42: -14.6589231, 2.1907566, -14.6795063, 2.1790240, -15.3227043, 15.2321930

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4786191
time: 30.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4806415
time: 36.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.8511353, 14.4232111, -14.8582096, 14.5176296, -29.3687649, 29.2814217
1: -8.3213196, 15.3521614, -8.3260422, 15.4242134, -23.4857178, 23.4221344
2: -7.1229086, 13.7586288, -7.1237144, 13.8314438, -19.6038666, 19.5718231
3: -10.5425234, 12.4740257, -10.5445604, 12.5409298, -21.3449936, 21.3241196
4: -8.0763483, 14.8806190, -8.0816727, 14.9599752, -20.9074707, 20.9117889
5: -10.6530018, 14.4652081, -10.6548786, 14.5483341, -23.6015930, 23.5388641
6: -18.7106819, 2.6698046, -18.7628651, 2.6752143, -18.9364357, 18.8003693
7: -13.6110144, 13.8677711, -13.6136227, 13.9483452, -25.7074814, 25.6318359
8: -15.1492119, 12.4612465, -15.1531601, 12.5379353, -26.2447281, 26.1946335
9: -9.8019943, 11.8728161, -9.8170576, 11.9331141, -21.3894043, 21.3195877
10: -21.2161369, 8.7998142, -21.2256813, 8.8641014, -28.7921524, 28.7171631
11: -22.3594379, 4.9557390, -22.3895817, 4.9652328, -24.7187424, 24.8517914
12: -22.4952564, 5.3668618, -22.5324783, 5.3795743, -25.7337570, 25.6574402
13: -17.6386642, 15.5503693, -17.6463051, 15.6363354, -32.7075958, 32.6181183
14: -39.4502869, -7.1562824, -39.4561119, -7.0809755, -26.8631210, 26.8264771
15: -11.9916058, 11.4582157, -12.0011053, 11.4866161, -21.0225143, 21.1330833
16: -20.1210003, 8.1354599, -20.1405411, 8.2131224, -28.3341217, 28.2760010
17: -40.6449699, -1.7937031, -40.6534882, -1.6828632, -38.9415283, 38.8597870
18: -16.6294632, 11.9961472, -16.6714916, 12.0091524, -28.6386147, 28.6676388
19: -15.8821402, 3.8632879, -15.9312916, 3.8647532, -18.8520203, 18.9211121
20: -13.3905478, 4.4948182, -13.4307508, 4.4973283, -17.8878765, 17.9255695
21: -19.0121212, 4.2678537, -19.0581322, 4.2717538, -22.6527328, 22.7838287
22: -15.5985317, 6.8100505, -15.6428556, 6.8152370, -21.8985062, 22.0423431
23: -15.0661354, 5.9845519, -15.1268177, 5.9881711, -19.7891235, 19.8776016
24: -11.5642996, 8.5756721, -11.6127262, 8.5841990, -18.5475922, 18.6337204
25: -13.8899994, 8.4316349, -13.9304810, 8.4370613, -21.3865433, 21.4587326
26: -23.3105907, 7.7525105, -23.3824463, 7.7570238, -31.0676155, 31.1349564
27: -14.6539583, 8.6972141, -14.7241316, 8.7025108, -23.2955627, 23.3857880
28: -15.3614693, 7.8435392, -15.4370451, 7.8454404, -21.1917496, 21.2731628
29: -17.4936123, 4.4202728, -17.5352173, 4.4240704, -21.0810776, 21.1211243
30: -19.3935051, 4.7327657, -19.4312286, 4.7399793, -21.0055466, 21.0520020
31: -17.0544930, 5.7788334, -17.1053391, 5.7869749, -22.8414688, 22.8841724
32: -14.4920406, 6.3034616, -14.5511875, 6.3087158, -19.4369354, 19.4406662
33: -24.2214584, 10.9072552, -24.2697659, 10.9159698, -32.8078613, 32.8386536
34: -23.4735241, 4.3479657, -23.5484085, 4.3506889, -23.2961960, 23.3710632
35: -19.5740738, 9.0775509, -19.6307793, 9.0820799, -26.6117859, 26.6284790
36: -18.4332790, 11.1508799, -18.5061760, 11.1544132, -27.2959518, 27.3671341
37: -29.1093826, 6.9299126, -29.1897068, 6.9328389, -32.8317566, 32.7548523
38: -22.7639008, 14.2588806, -22.8542881, 14.2647295, -34.6212616, 34.6838913
39: -28.0079918, 12.0020170, -28.0381260, 12.0078468, -39.5399475, 39.5684357
40: -25.7718449, 5.3476000, -25.8332119, 5.3503656, -29.0208969, 28.9529648
41: -14.7392521, 7.3610759, -14.8150806, 7.3634033, -19.2616272, 19.2300415
42: -14.6608496, 2.1961782, -14.7019958, 2.1996758, -15.3412628, 15.2624207

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058332, upper bound: 10.5029600
time: 29.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4806415
time: 36.96 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.8669920, 14.4451294, -14.7993908, 14.4442730, -29.3112640, 29.2445202
1: -8.3233719, 15.3644705, -8.2837915, 15.3634329, -23.0690689, 23.3865585
2: -7.1622601, 13.7991171, -7.0915356, 13.7968330, -19.6501999, 20.3175049
3: -10.5617981, 12.4951544, -10.5127048, 12.4921522, -21.3662033, 22.5130539
4: -8.1181507, 14.9210873, -8.0337992, 14.9203930, -20.9917068, 22.0387955
5: -10.6776266, 14.4942608, -10.6195621, 14.4898586, -23.5883560, 24.7091827
6: -18.7631073, 2.7244339, -18.7584057, 2.6534696, -18.9585266, 18.8356361
7: -13.6152267, 13.8851910, -13.5676498, 13.8826809, -25.6452942, 26.0870895
8: -15.1705093, 12.4926529, -15.1070347, 12.4916573, -26.2448578, 26.6729584
9: -9.8022709, 11.8785048, -9.7691927, 11.8728161, -20.9908447, 21.3036194
10: -21.2151585, 8.8178692, -21.1797428, 8.8112640, -28.7427902, 28.7174377
11: -22.3807011, 4.9769230, -22.3721714, 4.9550295, -25.0210953, 24.7386246
12: -22.5249996, 5.4028025, -22.5208130, 5.3516541, -26.1209412, 25.7818222
13: -17.6529579, 15.5748777, -17.6026917, 15.5681839, -32.6427460, 33.1775703
14: -39.4976196, -7.1076975, -39.4158401, -7.1121969, -26.9145737, 26.9335175
15: -12.0256443, 11.4884415, -11.9739256, 11.4870501, -21.1851501, 21.2138939
16: -20.1277809, 8.1501751, -20.0940628, 8.1371222, -28.2649040, 28.2442379
17: -40.7011185, -1.7348137, -40.6032372, -1.7362309, -38.9648895, 38.8684235
18: -16.6382103, 12.0019445, -16.6276150, 11.9820690, -28.6202793, 28.6295586
19: -15.8897095, 3.8608370, -15.8834152, 3.8403702, -18.8360214, 18.8082428
20: -13.3958473, 4.5001040, -13.3931122, 4.4753046, -17.8711510, 17.8932152
21: -19.0318680, 4.2831974, -19.0243988, 4.2505827, -22.7272034, 22.6872559
22: -15.6116495, 6.8089662, -15.5955429, 6.7919469, -21.9853516, 21.9106827
23: -15.0732517, 5.9834633, -15.0693283, 5.9572906, -20.3756104, 19.7906265
24: -11.5720539, 8.5774775, -11.5633802, 8.5553350, -18.8990021, 18.5844650
25: -13.8923454, 8.4329138, -13.8879938, 8.4055996, -21.5459442, 21.3816299
26: -23.3155613, 7.7429404, -23.3075562, 7.7189465, -31.0345078, 31.0504971
27: -14.6769428, 8.7104712, -14.6702366, 8.6665955, -23.3293533, 23.3460159
28: -15.3831139, 7.8534184, -15.3786325, 7.8068905, -21.9360962, 21.2237091
29: -17.5022354, 4.4184027, -17.4929619, 4.4013495, -21.1867676, 21.0780029
30: -19.4200363, 4.7615395, -19.4170761, 4.7191772, -21.3879547, 21.0519714
31: -17.0577621, 5.7793646, -17.0514107, 5.7541723, -22.8119354, 22.3937912
32: -14.5390911, 6.3442187, -14.5309906, 6.2859812, -19.4543076, 19.2480850
33: -24.2309170, 10.9172268, -24.2238064, 10.8793936, -32.7826996, 32.7802200
34: -23.4989662, 4.3585286, -23.4935913, 4.3098207, -23.4471207, 23.3222885
35: -19.5870838, 9.0749283, -19.5806808, 9.0420885, -26.5588379, 26.5772934
36: -18.4674072, 11.1623726, -18.4583092, 11.1097956, -27.2868347, 27.2904282
37: -29.1457367, 6.9646392, -29.1387882, 6.9035821, -33.3690186, 32.8964233
38: -22.7825966, 14.2469950, -22.7733879, 14.2178154, -34.8582611, 34.6172333
39: -28.0151901, 12.0057030, -28.0047264, 11.9890842, -39.5200348, 39.1384125
40: -25.8120899, 5.3893275, -25.8074875, 5.3358011, -29.0396881, 28.9077377
41: -14.7887583, 7.4080715, -14.7807570, 7.3342896, -19.5911942, 19.3535271
42: -14.7032614, 2.2436352, -14.6988659, 2.1800191, -15.9924850, 15.4309616

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5018409
time: 35.25 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5281700, upper bound: 10.5038766
time: 35.51 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.8854828, 14.4459038, -14.8597927, 14.5335674, -29.4190502, 29.3056965
1: -8.3363819, 15.3650169, -8.3269615, 15.4337473, -23.1529770, 23.4301529
2: -7.1730461, 13.7997780, -7.1260366, 13.8493614, -19.7142181, 20.3532486
3: -10.5727329, 12.4959812, -10.5482903, 12.5499725, -21.4372025, 22.5496292
4: -8.1329470, 14.9217100, -8.0851049, 14.9784117, -21.0641479, 22.0892563
5: -10.6897259, 14.4956074, -10.6590710, 14.5614080, -23.6743164, 24.7502594
6: -18.7642460, 2.7294078, -18.7876663, 2.6762748, -18.9838524, 18.8763199
7: -13.6306944, 13.8858986, -13.6161699, 13.9559803, -25.7373505, 26.1364670
8: -15.1851196, 12.4933567, -15.1568804, 12.5523319, -26.3199310, 26.7232208
9: -9.8161583, 11.8790522, -9.8179550, 11.9372244, -21.0698013, 21.3523865
10: -21.2289467, 8.8197317, -21.2276249, 8.8710232, -28.8246307, 28.7628937
11: -22.3834476, 4.9787350, -22.3991146, 4.9720907, -25.0428238, 24.7689438
12: -22.5265770, 5.4088101, -22.5465031, 5.3859038, -26.1559448, 25.8134308
13: -17.6663170, 15.5769930, -17.6493473, 15.6472912, -32.7366943, 33.2263412
14: -39.5100250, -7.1058407, -39.4581223, -7.0590563, -26.9841919, 26.9780502
15: -12.0317383, 11.4890776, -12.0033131, 11.5010462, -21.2028351, 21.2451553
16: -20.1426411, 8.1512089, -20.1477070, 8.2153645, -28.3580055, 28.2989159
17: -40.7165337, -1.7330742, -40.6585236, -1.6552200, -39.0613136, 38.9254494
18: -16.6399460, 12.0102940, -16.6729889, 12.0120888, -28.6520348, 28.6832829
19: -15.8914871, 3.8693295, -15.9347029, 3.8658237, -18.8630066, 18.8727493
20: -13.3972826, 4.5068765, -13.4333420, 4.4985776, -17.8958607, 17.9402180
21: -19.0343437, 4.2901306, -19.0670586, 4.2740312, -22.7522354, 22.7538300
22: -15.6142035, 6.8174944, -15.6461172, 6.8178658, -22.0123825, 21.9745636
23: -15.0748129, 5.9936028, -15.1299162, 5.9911442, -20.4109955, 19.8617401
24: -11.5730553, 8.5871887, -11.6139984, 8.5868702, -18.9327393, 18.6482544
25: -13.8936119, 8.4433088, -13.9311943, 8.4400921, -21.5822449, 21.4382401
26: -23.3181248, 7.7551188, -23.3845825, 7.7567444, -31.0748692, 31.1397018
27: -14.6787062, 8.7217064, -14.7331257, 8.7040863, -23.3691483, 23.4353485
28: -15.3843937, 7.8662386, -15.4461832, 7.8481169, -21.9798355, 21.3041458
29: -17.5048409, 4.4257507, -17.5375748, 4.4277706, -21.2161179, 21.1299210
30: -19.4208488, 4.7671714, -19.4430008, 4.7436476, -21.4168396, 21.0848007
31: -17.0596161, 5.7902384, -17.1136875, 5.7892556, -22.8488712, 22.4749222
32: -14.5409622, 6.3504367, -14.5735283, 6.3103118, -19.4813805, 19.3028297
33: -24.2336235, 10.9279690, -24.2758751, 10.9177427, -32.8216248, 32.8412552
34: -23.5004978, 4.3721604, -23.5590229, 4.3517895, -23.4907837, 23.4032211
35: -19.5886955, 9.0870457, -19.6358948, 9.0826025, -26.6002045, 26.6463547
36: -18.4694061, 11.1770725, -18.5210514, 11.1548309, -27.3328781, 27.3678360
37: -29.1484604, 6.9745016, -29.2065220, 6.9376812, -33.4106293, 32.9718933
38: -22.7850418, 14.2615938, -22.8612862, 14.2662420, -34.9090576, 34.7210083
39: -28.0183792, 12.0097961, -28.0488968, 12.0090828, -39.5439606, 39.1873627
40: -25.8147469, 5.3922167, -25.8539486, 5.3512807, -29.0611877, 28.9624329
41: -14.7911434, 7.4173946, -14.8365383, 7.3657489, -19.6227798, 19.4202538
42: -14.7051935, 2.2490566, -14.7213516, 2.2028158, -16.0131073, 15.4612732

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5018409
time: 35.16 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5038766
time: 33.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 70.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4786191
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4806415
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058332, upper bound: 10.5029600
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058332, upper bound: 10.4806415
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5018409
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5281700, upper bound: 10.5038766
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5018409
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.63
Output dim: 4, lower bound: -10.5058412, upper bound: 10.5038766

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.7608471, 14.4190674, -14.7642832, 14.4328117, -29.1936588, 29.1833496
1: -8.2561474, 15.3490019, -8.2585449, 15.3566713, -23.3521347, 23.3516006
2: -7.0681801, 13.7551003, -7.0702724, 13.7776041, -19.4947205, 19.5164490
3: -10.4899845, 12.4695387, -10.4917202, 12.4814653, -21.2299194, 21.2662125
4: -8.0052719, 14.8780251, -8.0063782, 14.9010611, -20.7776337, 20.8367538
5: -10.5943136, 14.4597921, -10.5962782, 14.4748535, -23.4663162, 23.4747543
6: -18.7054653, 2.6299319, -18.7339191, 2.6360664, -18.8891487, 18.7202950
7: -13.5320282, 13.8635168, -13.5367851, 13.8733444, -25.5495224, 25.5510483
8: -15.0728674, 12.4560375, -15.0773697, 12.4751911, -26.1062317, 26.1153488
9: -9.7364788, 11.8696060, -9.7442093, 11.8706827, -21.2586441, 21.2432022
10: -21.1504555, 8.7905674, -21.1537342, 8.8008928, -28.6543121, 28.6403503
11: -22.3463821, 4.9447927, -22.3578434, 4.9472065, -24.6814499, 24.8035889
12: -22.4879189, 5.3161654, -22.5040531, 5.3281965, -25.6726074, 25.5781097
13: -17.5876522, 15.5403252, -17.5828247, 15.5535288, -32.5644913, 32.5367813
14: -39.3913689, -7.1647120, -39.3924866, -7.1372004, -26.7415771, 26.7536469
15: -11.9567080, 11.4545813, -11.9591665, 11.4711838, -20.9734573, 21.0841751
16: -20.0518246, 8.1289816, -20.0617580, 8.1322737, -28.1840973, 28.1907387
17: -40.5835838, -1.8018646, -40.5801353, -1.7669468, -38.7864075, 38.7782707
18: -16.6212654, 11.9542532, -16.6230621, 11.9639015, -28.5851669, 28.5773163
19: -15.8728456, 3.8309956, -15.8771334, 3.8281784, -18.8031311, 18.8205948
20: -13.3829622, 4.4627547, -13.3876572, 4.4625001, -17.8454628, 17.8504124
21: -18.9998302, 4.2416978, -19.0108891, 4.2397399, -22.6027451, 22.6780090
22: -15.5852852, 6.7713680, -15.5883322, 6.7752643, -21.8425598, 21.9346237
23: -15.0595045, 5.9495554, -15.0638294, 5.9443893, -19.7381897, 19.7772980
24: -11.5582762, 8.5390568, -11.5597296, 8.5407896, -18.4915771, 18.5314102
25: -13.8818750, 8.3833647, -13.8840771, 8.3856010, -21.3232193, 21.3554764
26: -23.2984886, 7.6922760, -23.3009911, 7.6971502, -30.9956398, 30.9932671
27: -14.6463375, 8.6552143, -14.6584854, 8.6507530, -23.2284470, 23.2475586
28: -15.3549557, 7.7912407, -15.3670502, 7.7876654, -21.1272736, 21.1493835
29: -17.4807739, 4.3924761, -17.4858208, 4.3904934, -21.0337601, 21.0418396
30: -19.3879967, 4.7102914, -19.4030609, 4.7094021, -20.9627686, 20.9942627
31: -17.0441856, 5.7309217, -17.0462723, 5.7346182, -22.7788048, 22.7771950
32: -14.4842510, 6.2630806, -14.5081930, 6.2684555, -19.3874092, 19.3489380
33: -24.2091522, 10.8480196, -24.2148895, 10.8549080, -32.7358093, 32.7246475
34: -23.4669971, 4.2801275, -23.4806194, 4.2840114, -23.2223816, 23.2322922
35: -19.5658264, 9.0119686, -19.5724640, 9.0167799, -26.5384064, 26.5026169
36: -18.4234009, 11.0724592, -18.4404373, 11.0797224, -27.2121582, 27.2226868
37: -29.0972137, 6.8750200, -29.1175499, 6.8810606, -32.7666168, 32.6253967
38: -22.7509804, 14.1786823, -22.7614555, 14.1870537, -34.5312653, 34.5078964
39: -27.9929161, 11.9606714, -27.9960022, 11.9704561, -39.4845123, 39.4730072
40: -25.7613010, 5.3082700, -25.7847214, 5.3179350, -28.9746933, 28.8581161
41: -14.7307673, 7.3178782, -14.7564240, 7.3176894, -19.2103271, 19.1257553
42: -14.6539831, 2.1655989, -14.6771917, 2.1672947, -15.3006935, 15.2010345

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4560148
time: 31.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4758202
time: 30.46 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.8299713, 14.4815149, -14.7936363, 14.4341917, -29.2641640, 29.2751503
1: -8.3062038, 15.3999786, -8.2796936, 15.3577156, -23.4031219, 23.4240341
2: -7.1090627, 13.7947655, -7.0881639, 13.7787132, -19.5345078, 19.5738373
3: -10.5292978, 12.5097504, -10.5087862, 12.4828243, -21.2707443, 21.3240509
4: -8.0596666, 14.9153404, -8.0291576, 14.9018497, -20.8272324, 20.8962479
5: -10.6382542, 14.5070238, -10.6153889, 14.4764328, -23.5103073, 23.5412598
6: -18.7392311, 2.6649723, -18.7355270, 2.6501369, -18.9410858, 18.7525177
7: -13.5919266, 13.9204788, -13.5630751, 13.8747387, -25.6087952, 25.6348038
8: -15.1315899, 12.5064754, -15.1025324, 12.4769640, -26.1635208, 26.1904678
9: -9.7875881, 11.9121504, -9.7654409, 11.8716879, -21.3083115, 21.3075333
10: -21.2028961, 8.8380861, -21.1748772, 8.8038359, -28.7087784, 28.7083359
11: -22.3591270, 4.9580002, -22.3618374, 4.9507980, -24.6999283, 24.8282700
12: -22.5242214, 5.3639584, -22.5062294, 5.3464041, -25.7299881, 25.6223831
13: -17.6283245, 15.5733280, -17.5979843, 15.5566673, -32.6138458, 32.5886841
14: -39.4372787, -7.1271467, -39.4118156, -7.1346598, -26.7866516, 26.8140488
15: -11.9884281, 11.4618664, -11.9704666, 11.4724255, -21.0041351, 21.1020737
16: -20.1060009, 8.1852837, -20.0839138, 8.1344986, -28.2404995, 28.2691975
17: -40.6312332, -1.7496758, -40.5989685, -1.7643661, -38.8377228, 38.8492928
18: -16.6569252, 11.9880180, -16.6255932, 11.9774904, -28.6344147, 28.6136112
19: -15.8997660, 3.8535366, -15.8801308, 3.8377900, -18.8535309, 18.8438339
20: -13.4131699, 4.4865599, -13.3901234, 4.4723115, -17.8854809, 17.8766823
21: -19.0239773, 4.2605824, -19.0148849, 4.2472315, -22.6560669, 22.7021027
22: -15.6217480, 6.7997365, -15.5925722, 6.7874794, -21.9066696, 21.9637222
23: -15.0837622, 5.9744415, -15.0657730, 5.9542913, -19.7748489, 19.8043671
24: -11.5827236, 8.5648212, -11.5616436, 8.5515823, -18.5381012, 18.5601768
25: -13.9089546, 8.4196301, -13.8867521, 8.4008617, -21.3743668, 21.3940582
26: -23.3567238, 7.7372956, -23.3047733, 7.7166805, -31.0734043, 31.0420685
27: -14.6782379, 8.6864014, -14.6608219, 8.6631384, -23.2886124, 23.2859039
28: -15.3900900, 7.8290720, -15.3691101, 7.8035021, -21.1788177, 21.1896591
29: -17.5058537, 4.4127674, -17.4899006, 4.3987398, -21.0706711, 21.0652924
30: -19.3973579, 4.7288399, -19.4049454, 4.7156343, -20.9820786, 21.0198822
31: -17.0781841, 5.7664819, -17.0496559, 5.7495680, -22.8277512, 22.8161373
32: -14.5197639, 6.2974963, -14.5105486, 6.2823009, -19.4381561, 19.3817368
33: -24.2609692, 10.8946705, -24.2187119, 10.8747044, -32.8085327, 32.7731094
34: -23.5180569, 4.3311439, -23.4825764, 4.3062611, -23.2977676, 23.2781906
35: -19.6167450, 9.0626011, -19.5750828, 9.0386906, -26.6128311, 26.5517120
36: -18.4839382, 11.1321011, -18.4435806, 11.1057568, -27.2991257, 27.2821732
37: -29.1548195, 6.9172168, -29.1211090, 6.8999095, -32.8468018, 32.6702271
38: -22.8311844, 14.2406921, -22.7656212, 14.2138157, -34.6393280, 34.5698471
39: -28.0428181, 11.9959669, -28.0006733, 11.9855671, -39.5556793, 39.5133820
40: -25.8099632, 5.3437762, -25.7878342, 5.3327332, -29.0406799, 28.8914642
41: -14.7689714, 7.3503666, -14.7587576, 7.3315749, -19.2628479, 19.1554108
42: -14.6771288, 2.1901391, -14.6790438, 2.1772885, -15.3296356, 15.2232513

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4580472
time: 32.47 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4778836
time: 34.87 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.7794361, 14.4198551, -14.8248873, 14.5161037, -29.2955399, 29.2447433
1: -8.2693653, 15.3495331, -8.3019657, 15.4230213, -23.4328384, 23.3958206
2: -7.0789957, 13.7557917, -7.1032858, 13.8301163, -19.5593567, 19.5493622
3: -10.5009031, 12.4704380, -10.5250502, 12.5392990, -21.2998199, 21.2994308
4: -8.0202599, 14.8786659, -8.0557089, 14.9590626, -20.8527908, 20.8868942
5: -10.6063814, 14.4611568, -10.6330280, 14.5464268, -23.5525436, 23.5125656
6: -18.7065620, 2.6351175, -18.7609749, 2.6591754, -18.9172974, 18.7611694
7: -13.5474501, 13.8642483, -13.5837393, 13.9467373, -25.6398315, 25.5960083
8: -15.0874557, 12.4567747, -15.1244583, 12.5358715, -26.1813965, 26.1616516
9: -9.7502975, 11.8702240, -9.7929344, 11.9319458, -21.3326721, 21.2893372
10: -21.1640053, 8.7924633, -21.2013206, 8.8606892, -28.7302628, 28.6829758
11: -22.3492775, 4.9466810, -22.3849545, 4.9610109, -24.6706848, 24.8071518
12: -22.4894295, 5.3222475, -22.5298271, 5.3587503, -25.7058182, 25.6095352
13: -17.6010361, 15.5425310, -17.6288414, 15.6327639, -32.6818085, 32.6076736
14: -39.4038429, -7.1628647, -39.4340172, -7.0839424, -26.8105316, 26.7961121
15: -11.9631014, 11.4552221, -11.9880333, 11.4852161, -20.9684219, 21.0901337
16: -20.0667591, 8.1299877, -20.1152515, 8.2105961, -28.2773552, 28.2452393
17: -40.5994225, -1.8001614, -40.6324577, -1.6857815, -38.8993073, 38.8322983
18: -16.6230316, 11.9624348, -16.6685886, 11.9934311, -28.6164627, 28.6310234
19: -15.8746300, 3.8393602, -15.9278240, 3.8535943, -18.8318024, 18.8878822
20: -13.3844042, 4.4697499, -13.4279213, 4.4859424, -17.8703461, 17.8976707
21: -19.0023327, 4.2485771, -19.0535793, 4.2628884, -22.6336060, 22.7553864
22: -15.5879831, 6.7796850, -15.6380501, 6.8010426, -21.8609161, 21.9854889
23: -15.0610752, 5.9596062, -15.1244678, 5.9766464, -19.7814941, 19.8569336
24: -11.5593300, 8.5486565, -11.6104422, 8.5716476, -18.5289764, 18.6002426
25: -13.8832264, 8.3937111, -13.9273758, 8.4194050, -21.3598328, 21.4126663
26: -23.3010807, 7.7044477, -23.3780632, 7.7346840, -31.0357647, 31.0825119
27: -14.6481256, 8.6662521, -14.7214546, 8.6880951, -23.2799072, 23.3530884
28: -15.3562469, 7.8039236, -15.4346380, 7.8270426, -21.1741943, 21.2364349
29: -17.4835186, 4.3996496, -17.5306091, 4.4144411, -21.0577850, 21.0916672
30: -19.3888931, 4.7158966, -19.4290886, 4.7322125, -20.9805298, 21.0190506
31: -17.0460415, 5.7417011, -17.1014042, 5.7697244, -22.8157654, 22.8431053
32: -14.4860992, 6.2694216, -14.5484219, 6.2929420, -19.4153481, 19.4022560
33: -24.2119675, 10.8587914, -24.2653675, 10.8933945, -32.7721710, 32.7829895
34: -23.4685593, 4.2937975, -23.5461235, 4.3254843, -23.2634277, 23.3118668
35: -19.5674858, 9.0239792, -19.6277218, 9.0570879, -26.5760040, 26.5687485
36: -18.4254513, 11.0871735, -18.5025463, 11.1247349, -27.2572327, 27.2981644
37: -29.1000710, 6.8848505, -29.1854267, 6.9114451, -32.7953033, 32.6982880
38: -22.7534885, 14.1935158, -22.8493919, 14.2343817, -34.5784912, 34.6106567
39: -27.9961758, 11.9650698, -28.0325584, 11.9907084, -39.5060425, 39.5173950
40: -25.7639275, 5.3113999, -25.8295174, 5.3335686, -29.0008087, 28.9166412
41: -14.7331648, 7.3272028, -14.8123055, 7.3475704, -19.2419434, 19.1949043
42: -14.6558208, 2.1712515, -14.6996861, 2.1882396, -15.3354454, 15.2471848

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4803501
time: 36.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030468, upper bound: 10.5001343
time: 34.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.8485594, 14.4823322, -14.8542728, 14.5174732, -29.3660316, 29.3366051
1: -8.3193483, 15.4005184, -8.3232098, 15.4240713, -23.4837799, 23.4683151
2: -7.1199045, 13.7954330, -7.1212764, 13.8312893, -19.5993042, 19.6068344
3: -10.5402822, 12.5106230, -10.5422039, 12.5406904, -21.3406448, 21.3573761
4: -8.0745525, 14.9159832, -8.0785637, 14.9598370, -20.9033737, 20.9460907
5: -10.6503696, 14.5083714, -10.6522417, 14.5480728, -23.5966492, 23.5791626
6: -18.7404003, 2.6700573, -18.7625790, 2.6732540, -18.9692154, 18.7933464
7: -13.6073313, 13.9211979, -13.6100798, 13.9481153, -25.6985016, 25.6799622
8: -15.1463118, 12.5071621, -15.1497936, 12.5376587, -26.2389069, 26.2369537
9: -9.8014994, 11.9127350, -9.8142605, 11.9329433, -21.3821106, 21.3540497
10: -21.2167377, 8.8400040, -21.2227459, 8.8636494, -28.7844009, 28.7515106
11: -22.3619041, 4.9599447, -22.3888474, 4.9646664, -24.6902161, 24.8287506
12: -22.5258789, 5.3700690, -22.5321484, 5.3770742, -25.7637253, 25.6533508
13: -17.6418343, 15.5755014, -17.6441269, 15.6358366, -32.7301788, 32.6606750
14: -39.4497528, -7.1252289, -39.4534454, -7.0813694, -26.8556671, 26.8568039
15: -11.9947052, 11.4624596, -11.9993649, 11.4864311, -20.9996185, 21.1083527
16: -20.1208687, 8.1862812, -20.1374931, 8.2127666, -28.3336353, 28.3237743
17: -40.6463051, -1.7479439, -40.6509552, -1.6832275, -38.9494019, 38.9030113
18: -16.6586628, 11.9964848, -16.6711006, 12.0072536, -28.6659164, 28.6675854
19: -15.9015341, 3.8621092, -15.9308214, 3.8633871, -18.8832397, 18.9111671
20: -13.4146290, 4.4935446, -13.4304094, 4.4958448, -17.9104729, 17.9239540
21: -19.0264244, 4.2676783, -19.0575886, 4.2706556, -22.6896133, 22.7788315
22: -15.6243553, 6.8082891, -15.6422539, 6.8135223, -21.9219131, 22.0154648
23: -15.0853262, 5.9847031, -15.1264553, 5.9867406, -19.8198013, 19.8835144
24: -11.5837622, 8.5746803, -11.6123810, 8.5826492, -18.5765610, 18.6289101
25: -13.9102612, 8.4301596, -13.9300518, 8.4349098, -21.4113235, 21.4514465
26: -23.3593178, 7.7495165, -23.3818951, 7.7542906, -31.1136093, 31.1314125
27: -14.6800289, 8.6977501, -14.7237625, 8.7007523, -23.3427048, 23.3907471
28: -15.3914146, 7.8420372, -15.4367342, 7.8432059, -21.2255325, 21.2776108
29: -17.5085201, 4.4201899, -17.5346546, 4.4228792, -21.0945435, 21.1154022
30: -19.3982162, 4.7346849, -19.4309502, 4.7387056, -21.0008087, 21.0443726
31: -17.0800304, 5.7775006, -17.1048126, 5.7849250, -22.8649559, 22.8823128
32: -14.5216904, 6.3037596, -14.5507984, 6.3068209, -19.4663696, 19.4352493
33: -24.2637787, 10.9054241, -24.2691879, 10.9132919, -32.8449707, 32.8309708
34: -23.5196419, 4.3447523, -23.5481091, 4.3477144, -23.3388824, 23.3575745
35: -19.6184330, 9.0747395, -19.6303787, 9.0791645, -26.6506958, 26.6171722
36: -18.4859314, 11.1468687, -18.5056992, 11.1509142, -27.3442230, 27.3575897
37: -29.1576996, 6.9270782, -29.1890678, 6.9303203, -32.8756561, 32.7427444
38: -22.8337021, 14.2554417, -22.8536396, 14.2611628, -34.6866150, 34.6723480
39: -28.0461197, 12.0001287, -28.0373363, 12.0057583, -39.5752258, 39.5582275
40: -25.8126354, 5.3467355, -25.8327198, 5.3484087, -29.0670319, 28.9500732
41: -14.7714081, 7.3597159, -14.8146477, 7.3614979, -19.2946930, 19.2246704
42: -14.6790791, 2.1956909, -14.7016296, 2.1982081, -15.3627472, 15.2679710

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5254541, upper bound: 10.4824064
time: 32.09 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5254541, upper bound: 10.5022205
time: 33.98 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.7952042, 14.4417429, -14.7658768, 14.4427090, -29.2379131, 29.2076187
1: -8.2712383, 15.3618641, -8.2594833, 15.3622093, -23.0158234, 23.3596268
2: -7.1183100, 13.7962494, -7.0710263, 13.7955055, -19.6050568, 20.2941513
3: -10.5201950, 12.4915237, -10.4932241, 12.4904661, -21.3220444, 22.4882431
4: -8.0618639, 14.9191322, -8.0075397, 14.9195023, -20.9343262, 22.0101395
5: -10.6310577, 14.4901409, -10.5977612, 14.4879227, -23.5390930, 24.6828995
6: -18.7590446, 2.6895313, -18.7564774, 2.6371660, -18.9365845, 18.7959404
7: -13.5516586, 13.8816481, -13.5377903, 13.8810339, -25.5793457, 26.0533524
8: -15.1087456, 12.4881334, -15.0782080, 12.4895611, -26.1814117, 26.6394043
9: -9.7506428, 11.8758602, -9.7451115, 11.8715734, -20.9372787, 21.2759781
10: -21.1632156, 8.8104916, -21.1555634, 8.8077793, -28.6867752, 28.6860352
11: -22.3703995, 4.9677734, -22.3673515, 4.9507313, -25.0033722, 24.7206268
12: -22.5192451, 5.3581228, -22.5180435, 5.3308401, -26.0934448, 25.7340164
13: -17.6153564, 15.5669813, -17.5851212, 15.5644951, -32.5936127, 33.1521034
14: -39.4510727, -7.1142464, -39.3937416, -7.1152763, -26.8626251, 26.9039612
15: -11.9968472, 11.4854364, -11.9604816, 11.4856215, -21.1537323, 21.1942062
16: -20.0734863, 8.1446285, -20.0687294, 8.1345158, -28.2080021, 28.2133579
17: -40.6551704, -1.7412663, -40.5817680, -1.7392769, -38.9158936, 38.8404999
18: -16.6317406, 11.9684267, -16.6246128, 11.9664984, -28.5982399, 28.5930405
19: -15.8821793, 3.8370199, -15.8798885, 3.8292599, -18.8141327, 18.7725143
20: -13.3897057, 4.4747996, -13.3902130, 4.4635491, -17.8532543, 17.8650131
21: -19.0220299, 4.2639470, -19.0197945, 4.2416158, -22.7018738, 22.6479721
22: -15.6009474, 6.7787762, -15.5905304, 6.7778673, -21.9563675, 21.8672409
23: -15.0681591, 5.9586077, -15.0669308, 5.9456854, -20.3569641, 19.7614059
24: -11.5670261, 8.5505772, -11.5610027, 8.5427704, -18.8761139, 18.5459824
25: -13.8854809, 8.3950558, -13.8847713, 8.3879280, -21.5181656, 21.3349915
26: -23.3060036, 7.6948714, -23.3030930, 7.6964712, -31.0024757, 30.9979649
27: -14.6711140, 8.6797266, -14.6675234, 8.6522732, -23.3020248, 23.2970886
28: -15.3778534, 7.8139629, -15.3761721, 7.7884817, -21.9109039, 21.1803436
29: -17.4920158, 4.3979406, -17.4881935, 4.3918080, -21.1649857, 21.0506439
30: -19.4153519, 4.7446871, -19.4148788, 4.7113538, -21.3702087, 21.0270576
31: -17.0492821, 5.7423558, -17.0474396, 5.7368999, -22.7861824, 22.3422165
32: -14.5331688, 6.3100224, -14.5282040, 6.2700601, -19.4318390, 19.2090454
33: -24.2213345, 10.8686695, -24.2193527, 10.8566675, -32.7495117, 32.7263489
34: -23.4939842, 4.3043008, -23.4912205, 4.2845654, -23.4160080, 23.2644501
35: -19.5804310, 9.0214767, -19.5775681, 9.0171843, -26.5266724, 26.5204773
36: -18.4595947, 11.0986233, -18.4545975, 11.0801420, -27.2491150, 27.2224960
37: -29.1363144, 6.9195638, -29.1343937, 6.8821754, -33.3360596, 32.8423691
38: -22.7721844, 14.1813736, -22.7685127, 14.1872349, -34.8167725, 34.5450058
39: -28.0032444, 11.9685116, -27.9991226, 11.9717236, -39.4884796, 39.0935974
40: -25.8041763, 5.3528728, -25.8037758, 5.3188477, -29.0149918, 28.8675003
41: -14.7827044, 7.3741918, -14.7778740, 7.3184805, -19.5691452, 19.3159714
42: -14.6983213, 2.2184823, -14.6965189, 2.1682951, -15.9735527, 15.3997498

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030604, upper bound: 10.4792182
time: 34.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030604, upper bound: 10.4990521
time: 31.52 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.8643799, 14.5041952, -14.7952127, 14.4440956, -29.3084755, 29.2994080
1: -8.3212757, 15.4128342, -8.2806196, 15.3632679, -23.0660400, 23.4320755
2: -7.1591854, 13.8358850, -7.0889468, 13.7966480, -19.6448364, 20.3519745
3: -10.5595016, 12.5317020, -10.5103455, 12.4918737, -21.3629074, 22.5501251
4: -8.1162415, 14.9564333, -8.0303307, 14.9202414, -20.9839096, 22.0707703
5: -10.6749907, 14.5373354, -10.6168804, 14.4895353, -23.5830841, 24.7494278
6: -18.7928219, 2.7245841, -18.7580681, 2.6512251, -18.9884987, 18.8276291
7: -13.6115780, 13.9386425, -13.5640793, 13.8824272, -25.6386261, 26.1371002
8: -15.1674442, 12.5385895, -15.1033611, 12.4913521, -26.2386932, 26.7149124
9: -9.8017673, 11.9183769, -9.7663126, 11.8725615, -20.9861450, 21.3403091
10: -21.2156334, 8.8579912, -21.1766796, 8.8107281, -28.7412643, 28.7540207
11: -22.3831711, 4.9809933, -22.3713646, 4.9543505, -25.0241852, 24.7452469
12: -22.5555401, 5.4058962, -22.5202599, 5.3490801, -26.1517258, 25.7783203
13: -17.6560116, 15.5999107, -17.6002922, 15.5675812, -32.6429443, 33.2002029
14: -39.4970207, -7.0766945, -39.4131126, -7.1127434, -26.9077072, 26.9646683
15: -12.0285530, 11.4926853, -11.9717855, 11.4868469, -21.1844406, 21.2127304
16: -20.1276340, 8.2009659, -20.0909119, 8.1367302, -28.2643642, 28.2918777
17: -40.7027817, -1.6890888, -40.6005859, -1.7367020, -38.9660797, 38.9114990
18: -16.6674137, 12.0021915, -16.6271210, 11.9800386, -28.6474533, 28.6293125
19: -15.9091206, 3.8595629, -15.8828850, 3.8388615, -18.8645554, 18.7958260
20: -13.4198971, 4.4986029, -13.3927002, 4.4733377, -17.8932343, 17.8913040
21: -19.0461655, 4.2828865, -19.0237808, 4.2491398, -22.7560577, 22.6720428
22: -15.6374092, 6.8071752, -15.5947752, 6.7901182, -22.0204773, 21.8964386
23: -15.0924397, 5.9835019, -15.0688953, 5.9555759, -20.3976898, 19.7884750
24: -11.5914736, 8.5763531, -11.5629406, 8.5535564, -18.9250107, 18.5747261
25: -13.9125242, 8.4313326, -13.8874378, 8.4032011, -21.5710678, 21.3735580
26: -23.3642654, 7.7398701, -23.3069057, 7.7159681, -31.0802345, 31.0467758
27: -14.7030087, 8.7109184, -14.6698360, 8.6646633, -23.3624496, 23.3354416
28: -15.4130363, 7.8518081, -15.3782558, 7.8043737, -21.9646606, 21.2206116
29: -17.5171204, 4.4182415, -17.4922390, 4.4000406, -21.2051086, 21.0741043
30: -19.4246788, 4.7632380, -19.4167538, 4.7175598, -21.3916931, 21.0526581
31: -17.0832863, 5.7779083, -17.0508213, 5.7518358, -22.8351212, 22.3792267
32: -14.5686684, 6.3444519, -14.5305700, 6.2838869, -19.4825363, 19.2415466
33: -24.2731514, 10.9153318, -24.2231483, 10.8764420, -32.8222656, 32.7744827
34: -23.5450363, 4.3552904, -23.4931564, 4.3067946, -23.4911880, 23.3103638
35: -19.6313877, 9.0720863, -19.5802441, 9.0390892, -26.6010666, 26.5695190
36: -18.5200443, 11.1582527, -18.4577332, 11.1062145, -27.3360901, 27.2818222
37: -29.1939354, 6.9617815, -29.1379967, 6.9010181, -33.4157257, 32.8871765
38: -22.8523712, 14.2434406, -22.7726364, 14.2139921, -34.9247284, 34.6069641
39: -28.0531845, 12.0038157, -28.0038280, 11.9868279, -39.5596466, 39.1348877
40: -25.8528061, 5.3883572, -25.8068562, 5.3336496, -29.0810242, 28.9006500
41: -14.8208866, 7.4066763, -14.7802095, 7.3323545, -19.6219482, 19.3456154
42: -14.7214756, 2.2430155, -14.6983891, 2.1782935, -16.0078773, 15.4219856

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5254364, upper bound: 10.4812535
time: 36.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5011276
time: 32.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.8138466, 14.4425535, -14.8264542, 14.5319700, -29.3458176, 29.2690086
1: -8.2844677, 15.3624020, -8.3029156, 15.4326019, -23.1009369, 23.4037933
2: -7.1291766, 13.7969503, -7.1056547, 13.8480186, -19.6697235, 20.3298874
3: -10.5311184, 12.4924107, -10.5287800, 12.5483513, -21.3919678, 22.5223999
4: -8.0768518, 14.9197340, -8.0591469, 14.9774799, -21.0095215, 22.0623474
5: -10.6431351, 14.4914780, -10.6372232, 14.5595112, -23.6252975, 24.7229233
6: -18.7601280, 2.6947031, -18.7857780, 2.6602640, -18.9647522, 18.8373833
7: -13.5671196, 13.8823919, -13.5862732, 13.9543447, -25.6696320, 26.1011353
8: -15.1233406, 12.4888639, -15.1281385, 12.5502510, -26.2565994, 26.6896820
9: -9.7644958, 11.8764458, -9.7938337, 11.9360237, -21.0143051, 21.3222198
10: -21.1767769, 8.8123703, -21.2033195, 8.8675928, -28.7627640, 28.7288513
11: -22.3732910, 4.9696484, -22.3944607, 4.9678431, -25.0056000, 24.7242966
12: -22.5206966, 5.3641496, -22.5438404, 5.3650393, -26.1286011, 25.7654724
13: -17.6286831, 15.5692043, -17.6318321, 15.6436548, -32.7108994, 33.2010345
14: -39.4635048, -7.1124058, -39.4360352, -7.0620461, -26.9315643, 26.9473572
15: -12.0032177, 11.4860535, -11.9902611, 11.4996376, -21.1487274, 21.2039986
16: -20.0883675, 8.1457138, -20.1224365, 8.2128439, -28.3012123, 28.2681503
17: -40.6709862, -1.7395592, -40.6375427, -1.6581154, -39.0128708, 38.8979836
18: -16.6335182, 11.9766064, -16.6700897, 11.9963799, -28.6298981, 28.6466961
19: -15.8839512, 3.8454084, -15.9312077, 3.8546491, -18.8428116, 18.8399124
20: -13.3911533, 4.4817963, -13.4304867, 4.4871616, -17.8783150, 17.9122829
21: -19.0245266, 4.2708454, -19.0625095, 4.2651210, -22.7325363, 22.7253799
22: -15.6036301, 6.7871151, -15.6413107, 6.8036876, -21.9747467, 21.9186325
23: -15.0697489, 5.9686637, -15.1275768, 5.9796329, -20.3974762, 19.8410492
24: -11.5680676, 8.5601845, -11.6117525, 8.5743027, -18.9127426, 18.6147881
25: -13.8868198, 8.4053745, -13.9280739, 8.4224548, -21.5545044, 21.3922043
26: -23.3086052, 7.7070222, -23.3801727, 7.7343788, -31.0429840, 31.0871944
27: -14.6729364, 8.6907902, -14.7304325, 8.6896572, -23.3533096, 23.4026413
28: -15.3791676, 7.8266501, -15.4437532, 7.8297052, -21.9587555, 21.2673874
29: -17.4947433, 4.4051256, -17.5329685, 4.4181070, -21.1897888, 21.1005173
30: -19.4162502, 4.7503171, -19.4408779, 4.7359061, -21.3934174, 21.0517960
31: -17.0511475, 5.7531157, -17.1098022, 5.7719769, -22.8231239, 22.4265671
32: -14.5350046, 6.3163652, -14.5707645, 6.2945333, -19.4597626, 19.2651367
33: -24.2241669, 10.8794651, -24.2714615, 10.8950958, -32.7858582, 32.7856369
34: -23.4955273, 4.3179693, -23.5567513, 4.3265123, -23.4577408, 23.3440247
35: -19.5821381, 9.0334949, -19.6328201, 9.0576153, -26.5643845, 26.5866394
36: -18.4615974, 11.1133432, -18.5174465, 11.1251545, -27.2941360, 27.2989807
37: -29.1391487, 6.9293766, -29.2022572, 6.9162469, -33.3742523, 32.9152756
38: -22.7746658, 14.1961899, -22.8564949, 14.2358713, -34.8659363, 34.6477966
39: -28.0065231, 11.9728785, -28.0434074, 11.9919186, -39.5100403, 39.1414032
40: -25.8067570, 5.3559999, -25.8502693, 5.3344989, -29.0411224, 28.9264297
41: -14.7850904, 7.3835306, -14.8337383, 7.3499632, -19.6023865, 19.3851242
42: -14.7001600, 2.2241385, -14.7190361, 2.1913619, -16.0073586, 15.4459534

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5035489
time: 31.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643
time: 32.25 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.8829622, 14.5050402, -14.8558292, 14.5333681, -29.4163303, 29.3608704
1: -8.3344231, 15.4134121, -8.3240948, 15.4336357, -23.1512985, 23.4763107
2: -7.1700296, 13.8366194, -7.1235981, 13.8491764, -19.7096252, 20.3878174
3: -10.5704432, 12.5325890, -10.5459671, 12.5497322, -21.4328079, 22.5832062
4: -8.1311941, 14.9570503, -8.0820074, 14.9782810, -21.0600739, 22.1230927
5: -10.6870527, 14.5387115, -10.6564121, 14.5611248, -23.6693726, 24.7896500
6: -18.7939873, 2.7296619, -18.7874203, 2.6743670, -19.0166283, 18.8693085
7: -13.6269712, 13.9393864, -13.6126184, 13.9557819, -25.7283173, 26.1852264
8: -15.1821709, 12.5393133, -15.1534443, 12.5520439, -26.3141556, 26.7653427
9: -9.8156796, 11.9189520, -9.8151340, 11.9370298, -21.0629654, 21.3868866
10: -21.2295456, 8.8599243, -21.2247429, 8.8705864, -28.8168106, 28.7973022
11: -22.3859444, 4.9828701, -22.3983822, 4.9715586, -25.0271683, 24.7458954
12: -22.5571918, 5.4119654, -22.5461636, 5.3833666, -26.1868210, 25.8092651
13: -17.6695061, 15.6022015, -17.6471405, 15.6467953, -32.7592621, 33.2493439
14: -39.5094643, -7.0747910, -39.4554062, -7.0594521, -26.9767609, 27.0083771
15: -12.0348425, 11.4933157, -12.0015650, 11.5008678, -21.1799240, 21.2229614
16: -20.1424999, 8.2019987, -20.1446686, 8.2150269, -28.3575268, 28.3466682
17: -40.7178841, -1.6873951, -40.6560211, -1.6555691, -39.0623169, 38.9686279
18: -16.6691818, 12.0106392, -16.6726093, 12.0101910, -28.6793728, 28.6832485
19: -15.9108677, 3.8681531, -15.9342270, 3.8644691, -18.8942490, 18.8632812
20: -13.4213524, 4.5056014, -13.4329567, 4.4971046, -17.9184570, 17.9385586
21: -19.0486393, 4.2899771, -19.0664654, 4.2728729, -22.7893677, 22.7488403
22: -15.6400051, 6.8157229, -15.6455288, 6.8161192, -22.0357513, 21.9486923
23: -15.0939817, 5.9937253, -15.1295662, 5.9897170, -20.4400330, 19.8675919
24: -11.5924950, 8.5861855, -11.6136751, 8.5852985, -18.9625549, 18.6434669
25: -13.9138622, 8.4418163, -13.9307499, 8.4379425, -21.6075745, 21.4309616
26: -23.3668613, 7.7521648, -23.3840313, 7.7539897, -31.1208515, 31.1361961
27: -14.7048168, 8.7222853, -14.7327776, 8.7023296, -23.4071465, 23.4403076
28: -15.4143381, 7.8647509, -15.4458456, 7.8458381, -22.0137939, 21.3085556
29: -17.5197392, 4.4256353, -17.5369949, 4.4265795, -21.2291794, 21.1242447
30: -19.4255447, 4.7690711, -19.4427452, 4.7424135, -21.4181900, 21.0771294
31: -17.0851440, 5.7889233, -17.1131706, 5.7871909, -22.8723354, 22.4637222
32: -14.5705738, 6.3507247, -14.5731916, 6.3084068, -19.5107765, 19.2978249
33: -24.2760277, 10.9260855, -24.2752914, 10.9150286, -32.8586578, 32.8334579
34: -23.5465527, 4.3689423, -23.5587101, 4.3488264, -23.5330429, 23.3896866
35: -19.6330223, 9.0842285, -19.6354675, 9.0796862, -26.6390457, 26.6350861
36: -18.5220699, 11.1730442, -18.5205936, 11.1513739, -27.3811493, 27.3582077
37: -29.1967583, 6.9716034, -29.2058754, 6.9351959, -33.4539490, 32.9597092
38: -22.8549309, 14.2581091, -22.8606834, 14.2626657, -34.9738617, 34.7094727
39: -28.0564499, 12.0079708, -28.0481567, 12.0070400, -39.5792694, 39.1824188
40: -25.8555202, 5.3913298, -25.8534355, 5.3493147, -29.1073151, 28.9598312
41: -14.8233223, 7.4159932, -14.8361273, 7.3638368, -19.6553726, 19.4149017
42: -14.7234020, 2.2485621, -14.7209816, 2.2013609, -16.0384407, 15.4667816

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5056084
time: 37.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643
time: 37.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 78.08 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4560148
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4758202
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4580472
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4778836
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.4803501
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030468, upper bound: 10.5001343
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5254541, upper bound: 10.4824064
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5254541, upper bound: 10.5022205
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030604, upper bound: 10.4792182
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030604, upper bound: 10.4990521
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5254364, upper bound: 10.4812535
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5011276
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5035489
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5056084
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 78.08
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.7540274, 14.4183531, -14.7312965, 14.4293623, -29.1833897, 29.1496506
1: -8.2515287, 15.3482990, -8.2361240, 15.3533745, -23.3442230, 23.3284531
2: -7.0616655, 13.7544651, -7.0386953, 13.7744293, -19.4851532, 19.4840469
3: -10.4823465, 12.4686785, -10.4544706, 12.4770479, -21.2176208, 21.2274475
4: -7.9994411, 14.8777657, -7.9780378, 14.8996754, -20.7702255, 20.8079147
5: -10.5867996, 14.4588156, -10.5596628, 14.4700937, -23.4537277, 23.4365005
6: -18.7043629, 2.6282682, -18.7285042, 2.6279421, -18.8740082, 18.7127075
7: -13.5274363, 13.8626385, -13.5146322, 13.8692274, -25.5390854, 25.5209579
8: -15.0668774, 12.4544163, -15.0484390, 12.4672222, -26.0912857, 26.0796585
9: -9.7349529, 11.8681021, -9.7368612, 11.8633747, -21.2485657, 21.2338943
10: -21.1485138, 8.7775192, -21.1441460, 8.7373924, -28.5878143, 28.6170197
11: -22.3453522, 4.9310074, -22.3526535, 4.8799143, -24.6125717, 24.7847519
12: -22.4861717, 5.3046069, -22.4955635, 5.2717986, -25.6144333, 25.5587234
13: -17.5808945, 15.5383644, -17.5511856, 15.5439720, -32.5479736, 32.5022736
14: -39.3880424, -7.1736498, -39.3763809, -7.1808491, -26.6970139, 26.7279053
15: -11.9521847, 11.4536972, -11.9371939, 11.4668169, -20.9644241, 21.0614281
16: -20.0503654, 8.1235886, -20.0544186, 8.1066370, -28.1570015, 28.1780071
17: -40.5815277, -1.8158302, -40.5700760, -1.8349495, -38.7176514, 38.7542458
18: -16.6195126, 11.9453869, -16.6146679, 11.9205446, -28.5400581, 28.5600548
19: -15.8713398, 3.8240814, -15.8697090, 3.7944384, -18.7672653, 18.8061943
20: -13.3815174, 4.4579492, -13.3805141, 4.4390993, -17.8206177, 17.8384628
21: -18.9982681, 4.2320271, -19.0032539, 4.1925716, -22.5539932, 22.6612015
22: -15.5834017, 6.7662272, -15.5790367, 6.7500639, -21.8087311, 21.9188995
23: -15.0582876, 5.9435930, -15.0578737, 5.9157438, -19.7091141, 19.7648621
24: -11.5570583, 8.5348063, -11.5537128, 8.5199986, -18.4727554, 18.5217514
25: -13.8804722, 8.3785858, -13.8772402, 8.3622856, -21.2984695, 21.3439331
26: -23.2965202, 7.6799741, -23.2914104, 7.6371355, -30.9336548, 30.9713840
27: -14.6448298, 8.6528139, -14.6511621, 8.6390648, -23.2163239, 23.2382202
28: -15.3536053, 7.7879906, -15.3604927, 7.7719564, -21.1098251, 21.1393661
29: -17.4794827, 4.3842545, -17.4794807, 4.3502436, -20.9926910, 21.0276871
30: -19.3871574, 4.7021308, -19.3989105, 4.6696248, -20.9213867, 20.9816818
31: -17.0422401, 5.7234106, -17.0368595, 5.6979446, -22.7401848, 22.7602692
32: -14.4826870, 6.2620821, -14.5006018, 6.2636833, -19.3695717, 19.3387260
33: -24.2020874, 10.8461876, -24.1815605, 10.8460855, -32.7207947, 32.6917496
34: -23.4625511, 4.2788115, -23.4595642, 4.2775211, -23.2111282, 23.2109451
35: -19.5597916, 9.0105734, -19.5429001, 9.0098877, -26.5250854, 26.4686279
36: -18.4184837, 11.0715075, -18.4166164, 11.0749483, -27.2029648, 27.2016373
37: -29.0956612, 6.8729181, -29.1100349, 6.8710752, -32.7541199, 32.6148453
38: -22.7449455, 14.1779842, -22.7324219, 14.1836901, -34.5223236, 34.4810104
39: -27.9898777, 11.9595919, -27.9814186, 11.9652290, -39.4759369, 39.4573212
40: -25.7591228, 5.3070993, -25.7738762, 5.3120890, -28.9655685, 28.8461685
41: -14.7295027, 7.3160219, -14.7502317, 7.3085976, -19.1967545, 19.1145401
42: -14.6529789, 2.1623528, -14.6723614, 2.1515732, -15.2690697, 15.1903267

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4525027
time: 31.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4525027
time: 31.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.7559090, 14.4184189, -14.7786732, 14.5055332, -29.2614422, 29.1970921
1: -8.2526150, 15.3483915, -8.2662716, 15.4076834, -23.4000626, 23.3588791
2: -7.0635586, 13.7541428, -7.0750360, 13.8787680, -19.5911942, 19.5175095
3: -10.4847546, 12.4679947, -10.4938641, 12.6000729, -21.3429642, 21.2647400
4: -8.0003681, 14.8776541, -8.0154438, 14.9893818, -20.8622513, 20.8387680
5: -10.5892286, 14.4581709, -10.6021442, 14.5946913, -23.5802917, 23.4772491
6: -18.7043037, 2.6210284, -18.7477188, 2.6346989, -18.8843307, 18.7593613
7: -13.5287447, 13.8620872, -13.5502853, 13.9215355, -25.6176071, 25.5443039
8: -15.0684710, 12.4542322, -15.0788460, 12.5671139, -26.2150726, 26.1018524
9: -9.7342386, 11.8667660, -9.7702866, 11.8817186, -21.2646866, 21.2712326
10: -21.1485138, 8.7808456, -21.3480473, 8.8117390, -28.6574020, 28.8246460
11: -22.3452778, 4.9358768, -22.5663376, 4.9441128, -24.6712875, 25.0041656
12: -22.4857941, 5.3084297, -22.6712112, 5.3357668, -25.6765900, 25.7384567
13: -17.5702438, 15.5386086, -17.5698643, 15.6585274, -32.6605682, 32.5246887
14: -39.3879700, -7.1723385, -39.5360107, -7.1395340, -26.7416153, 26.8961716
15: -11.9354992, 11.4537067, -11.9387741, 11.5177984, -21.0116348, 21.0801849
16: -20.0500126, 8.1082516, -20.1459675, 8.1122122, -28.1622238, 28.2542191
17: -40.5814323, -1.8107224, -40.7661743, -1.7703991, -38.7820435, 38.9554520
18: -16.6194744, 11.9481487, -16.7797775, 11.9584217, -28.5778961, 28.7279263
19: -15.8712673, 3.8263297, -16.0045242, 3.8246350, -18.7985229, 18.9433670
20: -13.3818092, 4.4591265, -13.4812250, 4.4631577, -17.8449669, 17.9403515
21: -18.9984665, 4.2353058, -19.1750679, 4.2350798, -22.5956116, 22.8369980
22: -15.5828285, 6.7678900, -15.6576977, 6.7771683, -21.8241119, 22.0363388
23: -15.0584164, 5.9451208, -15.1588202, 5.9457369, -19.7382278, 19.8664322
24: -11.5565395, 8.5360003, -11.6273298, 8.5402527, -18.4853439, 18.5863190
25: -13.8798065, 8.3797226, -13.9534664, 8.3863792, -21.3226547, 21.4225845
26: -23.2958870, 7.6845493, -23.4903908, 7.6962609, -30.9921474, 31.1749401
27: -14.6447830, 8.6479721, -14.7124548, 8.6439533, -23.2269135, 23.3080597
28: -15.3536959, 7.7883978, -15.4441586, 7.7929811, -21.1306381, 21.2228012
29: -17.4791260, 4.3867149, -17.5705109, 4.3870497, -21.0287704, 21.1213455
30: -19.3870525, 4.7054238, -19.5155602, 4.7177382, -20.9652557, 21.1017303
31: -17.0425682, 5.7258821, -17.1880474, 5.7305164, -22.7730846, 22.9139290
32: -14.4822359, 6.2610879, -14.5215282, 6.2807937, -19.3678894, 19.4117126
33: -24.2044907, 10.8463364, -24.2266464, 10.9604816, -32.8448944, 32.7317352
34: -23.4620171, 4.2783275, -23.4829445, 4.3322411, -23.2722626, 23.2318192
35: -19.5572968, 9.0109634, -19.5693703, 9.1071529, -26.6441956, 26.5005341
36: -18.4146576, 11.0715446, -18.4369678, 11.1396990, -27.2482910, 27.2157669
37: -29.0942497, 6.8700686, -29.1470413, 6.8836102, -32.7790222, 32.6470108
38: -22.7436142, 14.1775742, -22.7728844, 14.2664280, -34.5986023, 34.5209503
39: -27.9822121, 11.9596243, -27.9947071, 12.0448008, -39.5659180, 39.4789124
40: -25.7590084, 5.3017025, -25.8046684, 5.3292437, -28.9837189, 28.8730621
41: -14.7292032, 7.3156142, -14.7663593, 7.3352766, -19.2168045, 19.1459160
42: -14.6531410, 2.1532154, -14.7177362, 2.1688709, -15.2968445, 15.2856140

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4730571
time: 33.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4730571
time: 34.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.8231812, 14.4807949, -14.7606192, 14.4307251, -29.2539062, 29.2414131
1: -8.3015566, 15.3993025, -8.2572851, 15.3543825, -23.3951492, 23.4008408
2: -7.1025777, 13.7941027, -7.0565844, 13.7755842, -19.5249176, 19.5414886
3: -10.5216341, 12.5088167, -10.4715548, 12.4784508, -21.2584686, 21.2852478
4: -8.0538187, 14.9150572, -8.0008116, 14.9004545, -20.8198471, 20.8674011
5: -10.6307287, 14.5060205, -10.5788012, 14.4717073, -23.4977264, 23.5030289
6: -18.7381210, 2.6633015, -18.7301044, 2.6419926, -18.9259529, 18.7449074
7: -13.5873222, 13.9195862, -13.5409336, 13.8706160, -25.5983658, 25.6047592
8: -15.1256447, 12.5048599, -15.0736151, 12.4690180, -26.1485825, 26.1547546
9: -9.7860718, 11.9106236, -9.7581015, 11.8643589, -21.2981949, 21.2982483
10: -21.2009087, 8.8250446, -21.1652908, 8.7403641, -28.6422501, 28.6850052
11: -22.3580799, 4.9442282, -22.3566208, 4.8835239, -24.6310349, 24.8093948
12: -22.5224819, 5.3523688, -22.4977531, 5.2900338, -25.6718597, 25.6030121
13: -17.6215401, 15.5713882, -17.5663338, 15.5471182, -32.5972748, 32.5541763
14: -39.4340210, -7.1361055, -39.3956871, -7.1783257, -26.7420883, 26.7883148
15: -11.9839134, 11.4609642, -11.9484997, 11.4680309, -20.9950562, 21.0793495
16: -20.1045094, 8.1799250, -20.0765514, 8.1088486, -28.2133579, 28.2564774
17: -40.6291656, -1.7635994, -40.5889359, -1.8323765, -38.7689362, 38.8253365
18: -16.6552162, 11.9791584, -16.6171913, 11.9341087, -28.5893250, 28.5963497
19: -15.8982544, 3.8466063, -15.8726768, 3.8040247, -18.8176804, 18.8293991
20: -13.4117203, 4.4817600, -13.3829594, 4.4489160, -17.8606358, 17.8647194
21: -19.0224056, 4.2509146, -19.0072746, 4.2000866, -22.6073303, 22.6852417
22: -15.6198444, 6.7945571, -15.5832634, 6.7622766, -21.8728333, 21.9479294
23: -15.0825748, 5.9685173, -15.0598526, 5.9256649, -19.7457657, 19.7919235
24: -11.5815287, 8.5605793, -11.5556288, 8.5307817, -18.5192795, 18.5505066
25: -13.9075451, 8.4148445, -13.8799114, 8.3775568, -21.3495865, 21.3824997
26: -23.3547993, 7.7250042, -23.2952080, 7.6566625, -31.0114613, 31.0202122
27: -14.6767082, 8.6839790, -14.6534805, 8.6514301, -23.2764511, 23.2765274
28: -15.3887587, 7.8258238, -15.3625526, 7.7878399, -21.1613846, 21.1796417
29: -17.5045776, 4.4045048, -17.4835567, 4.3584871, -21.0295639, 21.0511398
30: -19.3965187, 4.7206779, -19.4008026, 4.6758766, -20.9406967, 21.0072823
31: -17.0762215, 5.7589374, -17.0402260, 5.7129021, -22.7891235, 22.7991638
32: -14.5182123, 6.2964892, -14.5029583, 6.2775164, -19.4203224, 19.3715172
33: -24.2539654, 10.8928432, -24.1853828, 10.8659182, -32.7934875, 32.7401428
34: -23.5136108, 4.3297949, -23.4614697, 4.2998013, -23.2865601, 23.2568283
35: -19.6106834, 9.0611715, -19.5455170, 9.0318251, -26.5994568, 26.5176849
36: -18.4789810, 11.1310863, -18.4197598, 11.1010113, -27.2899628, 27.2611847
37: -29.1533146, 6.9151716, -29.1136589, 6.8898711, -32.8343048, 32.6596909
38: -22.8251972, 14.2400408, -22.7365799, 14.2104607, -34.6303864, 34.5430145
39: -28.0398350, 11.9948454, -27.9861298, 11.9803038, -39.5471344, 39.4977264
40: -25.8076973, 5.3425455, -25.7769928, 5.3269739, -29.0316162, 28.8795242
41: -14.7676773, 7.3484774, -14.7525501, 7.3224907, -19.2492676, 19.1441727
42: -14.6761408, 2.1868947, -14.6742296, 2.1615562, -15.2980194, 15.2125359

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4545472
time: 32.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5226207, upper bound: 10.4545472
time: 43.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.8250895, 14.4809036, -14.8080320, 14.5069160, -29.3320045, 29.2889366
1: -8.3026600, 15.3993788, -8.2874060, 15.4087019, -23.4510574, 23.4313202
2: -7.1044412, 13.7938156, -7.0929294, 13.8798981, -19.6309204, 19.5749435
3: -10.5240116, 12.5081673, -10.5109863, 12.6014671, -21.3838043, 21.3225403
4: -8.0547523, 14.9149876, -8.0382290, 14.9901447, -20.9118423, 20.8982315
5: -10.6331654, 14.5053701, -10.6212730, 14.5963259, -23.6242294, 23.5438080
6: -18.7380924, 2.6560555, -18.7493324, 2.6487632, -18.9362793, 18.7915459
7: -13.5886288, 13.9190731, -13.5765686, 13.9228945, -25.6768646, 25.6281204
8: -15.1271887, 12.5046673, -15.1040344, 12.5689373, -26.2723846, 26.1769638
9: -9.7853413, 11.9092731, -9.7914772, 11.8827219, -21.3143234, 21.3355408
10: -21.2009773, 8.8283577, -21.3692017, 8.8147039, -28.7118607, 28.8926773
11: -22.3580170, 4.9491415, -22.5703392, 4.9477367, -24.6897354, 25.0288086
12: -22.5221367, 5.3562546, -22.6734581, 5.3540545, -25.7340240, 25.7827606
13: -17.6108685, 15.5716343, -17.5850830, 15.6616888, -32.7098846, 32.5765915
14: -39.4339027, -7.1347055, -39.5553665, -7.1370411, -26.7867126, 26.9565887
15: -11.9672232, 11.4609842, -11.9500589, 11.5190010, -21.0422821, 21.0981026
16: -20.1041565, 8.1645775, -20.1681137, 8.1144056, -28.2185631, 28.3326912
17: -40.6290283, -1.7585297, -40.7850227, -1.7677822, -38.8333130, 39.0264931
18: -16.6551514, 11.9819212, -16.7822952, 11.9720421, -28.6271935, 28.7642174
19: -15.8981924, 3.8488631, -16.0075417, 3.8342366, -18.8489609, 18.9666138
20: -13.4120188, 4.4829240, -13.4836655, 4.4729729, -17.8849907, 17.9665890
21: -19.0226669, 4.2541900, -19.1790314, 4.2426233, -22.6489563, 22.8610611
22: -15.6193104, 6.7962418, -15.6619453, 6.7893896, -21.8881912, 22.0653915
23: -15.0826836, 5.9700146, -15.1607838, 5.9556527, -19.7748871, 19.8935089
24: -11.5810108, 8.5617619, -11.6292610, 8.5510292, -18.5318909, 18.6150818
25: -13.9068680, 8.4159698, -13.9561100, 8.4016352, -21.3738174, 21.4611816
26: -23.3541393, 7.7296047, -23.4942360, 7.7157402, -31.0698795, 31.2238407
27: -14.6767197, 8.6792135, -14.7147999, 8.6563435, -23.2870636, 23.3463745
28: -15.3888445, 7.8262539, -15.4462166, 7.8088813, -21.1821976, 21.2630692
29: -17.5042362, 4.4070034, -17.5745716, 4.3953199, -21.0656738, 21.1447830
30: -19.3964081, 4.7240100, -19.5174313, 4.7240267, -20.9845657, 21.1273651
31: -17.0765533, 5.7614307, -17.1914101, 5.7454362, -22.8219891, 22.9528408
32: -14.5177326, 6.2955022, -14.5238543, 6.2946401, -19.4186325, 19.4445267
33: -24.2563362, 10.8929672, -24.2304115, 10.9803104, -32.9175415, 32.7801743
34: -23.5130653, 4.3292990, -23.4849014, 4.3544788, -23.3476715, 23.2777023
35: -19.6082115, 9.0615673, -19.5719376, 9.1290445, -26.7185364, 26.5495758
36: -18.4751530, 11.1311722, -18.4401417, 11.1657991, -27.3352509, 27.2752457
37: -29.1518250, 6.9122286, -29.1506386, 6.9024987, -32.8592224, 32.6918259
38: -22.8238373, 14.2396469, -22.7770691, 14.2931461, -34.7066650, 34.5829773
39: -28.0321617, 11.9948797, -27.9994011, 12.0598869, -39.6371460, 39.5193024
40: -25.8076363, 5.3371801, -25.8077946, 5.3440561, -29.0496979, 28.9064484
41: -14.7673979, 7.3481121, -14.7686949, 7.3491621, -19.2693329, 19.1755447
42: -14.6763039, 2.1777723, -14.7195845, 2.1788373, -15.3257790, 15.3077965

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4751287
time: 37.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4751287
time: 31.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.7726622, 14.4191704, -14.7918129, 14.5126591, -29.2853203, 29.2109833
1: -8.2647648, 15.3488579, -8.2795744, 15.4197216, -23.4248505, 23.3726273
2: -7.0725203, 13.7551346, -7.0717263, 13.8269882, -19.5497513, 19.5169983
3: -10.4932728, 12.4695168, -10.4878073, 12.5349245, -21.2875366, 21.2606125
4: -8.0144176, 14.8783817, -8.0273247, 14.9576874, -20.8453903, 20.8580551
5: -10.5988913, 14.4601879, -10.5964184, 14.5417137, -23.5399551, 23.4743042
6: -18.7054310, 2.6334438, -18.7555752, 2.6510286, -18.9021835, 18.7535553
7: -13.5428925, 13.8633995, -13.5615940, 13.9426069, -25.6293640, 25.5659332
8: -15.0814905, 12.4551601, -15.0954857, 12.5279560, -26.1664810, 26.1259003
9: -9.7487888, 11.8686724, -9.7855911, 11.9245911, -21.3225937, 21.2800446
10: -21.1620407, 8.7794170, -21.1916809, 8.7972078, -28.6637268, 28.6596527
11: -22.3481979, 4.9328747, -22.3797455, 4.8937263, -24.6018372, 24.7882767
12: -22.4876652, 5.3106413, -22.5213242, 5.3023586, -25.6476440, 25.5901413
13: -17.5942249, 15.5406017, -17.5971375, 15.6232004, -32.6652527, 32.5731125
14: -39.4005280, -7.1717978, -39.4178696, -7.1276159, -26.7659912, 26.7703476
15: -11.9585819, 11.4543219, -11.9660683, 11.4808264, -20.9593887, 21.0673828
16: -20.0652542, 8.1246529, -20.1079140, 8.1849833, -28.2502365, 28.2325668
17: -40.5973358, -1.8141251, -40.6224251, -1.7538376, -38.8304901, 38.8083000
18: -16.6213112, 11.9535789, -16.6601486, 11.9500933, -28.5714035, 28.6137276
19: -15.8730955, 3.8324299, -15.9203815, 3.8198562, -18.7959290, 18.8734550
20: -13.3829441, 4.4649754, -13.4207821, 4.4625492, -17.8454933, 17.8857574
21: -19.0007839, 4.2388964, -19.0459614, 4.2157140, -22.5848846, 22.7385178
22: -15.5860748, 6.7745299, -15.6287498, 6.7758694, -21.8270798, 21.9697723
23: -15.0598640, 5.9536953, -15.1185436, 5.9480247, -19.7524261, 19.8445358
24: -11.5580902, 8.5443954, -11.6044483, 8.5508270, -18.5101624, 18.5905876
25: -13.8818016, 8.3889084, -13.9205360, 8.3961258, -21.3350601, 21.4011536
26: -23.2991238, 7.6921291, -23.3684559, 7.6746755, -30.9737988, 31.0605850
27: -14.6466303, 8.6638412, -14.7140846, 8.6763706, -23.2677689, 23.3437347
28: -15.3548965, 7.8007011, -15.4280586, 7.8113251, -21.1567459, 21.2264404
29: -17.4822159, 4.3914075, -17.5242767, 4.3741884, -21.0167007, 21.0775528
30: -19.3880310, 4.7077336, -19.4249611, 4.6924620, -20.9391479, 21.0064240
31: -17.0440845, 5.7341843, -17.0920067, 5.7330394, -22.7771244, 22.8261909
32: -14.4845448, 6.2684102, -14.5408487, 6.2881918, -19.3975563, 19.3920364
33: -24.2049103, 10.8569889, -24.2320633, 10.8845863, -32.7571259, 32.7500534
34: -23.4640675, 4.2924490, -23.5250435, 4.3190002, -23.2521820, 23.2905273
35: -19.5614262, 9.0226135, -19.5981350, 9.0501823, -26.5626526, 26.5347748
36: -18.4205627, 11.0861435, -18.4786892, 11.1199684, -27.2479706, 27.2771378
37: -29.0985317, 6.8827820, -29.1779137, 6.9014120, -32.7827301, 32.6877441
38: -22.7474785, 14.1928043, -22.8204422, 14.2309942, -34.5694733, 34.5837402
39: -27.9931488, 11.9639778, -28.0180283, 11.9854507, -39.4974823, 39.5017242
40: -25.7616997, 5.3101373, -25.8186722, 5.3277979, -28.9916687, 28.9046936
41: -14.7318945, 7.3253284, -14.8060913, 7.3384943, -19.2283401, 19.1836243
42: -14.6548233, 2.1680286, -14.6948843, 2.1725101, -15.3038368, 15.2364883

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4768374
time: 29.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4768374
time: 31.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.7745724, 14.4192743, -14.8392849, 14.5888548, -29.3634262, 29.2585602
1: -8.2658377, 15.3489122, -8.3097172, 15.4740467, -23.4807358, 23.4030838
2: -7.0744228, 13.7548513, -7.1080880, 13.9313412, -19.6558151, 19.5504532
3: -10.4956303, 12.4688797, -10.5272217, 12.6579075, -21.4128952, 21.2979202
4: -8.0153379, 14.8782721, -8.0647764, 15.0473881, -20.9374161, 20.8888931
5: -10.6013002, 14.4595280, -10.6388874, 14.6663084, -23.6664963, 23.5150757
6: -18.7053795, 2.6262178, -18.7747498, 2.6577878, -18.9125214, 18.8002052
7: -13.5441933, 13.8628349, -13.5972786, 13.9949074, -25.7079010, 25.5892715
8: -15.0830784, 12.4549608, -15.1259270, 12.6278210, -26.2902603, 26.1480942
9: -9.7480793, 11.8673410, -9.8190269, 11.9429674, -21.3386688, 21.3173599
10: -21.1620598, 8.7827749, -21.3955898, 8.8715954, -28.7333069, 28.8673477
11: -22.3481617, 4.9377937, -22.5934010, 4.9579182, -24.6605225, 25.0076370
12: -22.4872971, 5.3144851, -22.6970310, 5.3663597, -25.7098083, 25.7698822
13: -17.5835991, 15.5408764, -17.6158752, 15.7377625, -32.7778549, 32.5955658
14: -39.4004288, -7.1704273, -39.5774841, -7.0863180, -26.8105545, 26.9385910
15: -11.9418917, 11.4543324, -11.9676552, 11.5317993, -21.0066071, 21.0861893
16: -20.0649242, 8.1093521, -20.1994610, 8.1905670, -28.2554913, 28.3088131
17: -40.5971985, -1.8090210, -40.8184586, -1.6892776, -38.8948669, 39.0094376
18: -16.6212692, 11.9563532, -16.8252563, 11.9879580, -28.6092262, 28.7816086
19: -15.8730259, 3.8346949, -16.0551567, 3.8500433, -18.8271942, 19.0106430
20: -13.3832741, 4.4660993, -13.5214539, 4.4866028, -17.8698769, 17.9875526
21: -19.0010147, 4.2421622, -19.2177467, 4.2582431, -22.6264801, 22.9143219
22: -15.5855207, 6.7762151, -15.7073545, 6.8029966, -21.8424225, 22.0871964
23: -15.0599785, 5.9551783, -15.2194357, 5.9779983, -19.7815170, 19.9460526
24: -11.5575886, 8.5455856, -11.6780491, 8.5710735, -18.5227432, 18.6550865
25: -13.8811512, 8.3900394, -13.9967384, 8.4201889, -21.3592987, 21.4797745
26: -23.2984676, 7.6967039, -23.5674458, 7.7336884, -31.0321560, 31.2641487
27: -14.6466236, 8.6590652, -14.7753601, 8.6812477, -23.2783737, 23.4134979
28: -15.3549671, 7.8011150, -15.5116978, 7.8323536, -21.1775742, 21.3098373
29: -17.4818630, 4.3938837, -17.6152496, 4.4110265, -21.0527878, 21.1711578
30: -19.3879547, 4.7110271, -19.5415688, 4.7405591, -20.9830017, 21.1264763
31: -17.0444031, 5.7366400, -17.2431583, 5.7655730, -22.8099766, 22.9797974
32: -14.4840622, 6.2674203, -14.5617809, 6.3052931, -19.3958549, 19.4650307
33: -24.2073021, 10.8571243, -24.2772064, 10.9989891, -32.8812256, 32.7901306
34: -23.4635983, 4.2919822, -23.5484695, 4.3736396, -23.3133011, 23.3114014
35: -19.5589638, 9.0230055, -19.6245804, 9.1474190, -26.6817017, 26.5666656
36: -18.4166965, 11.0862293, -18.4991360, 11.1847076, -27.2933197, 27.2912292
37: -29.0970707, 6.8798428, -29.2149601, 6.9139843, -32.8076477, 32.7199326
38: -22.7461395, 14.1924229, -22.8609257, 14.3137054, -34.6457977, 34.6236572
39: -27.9854164, 11.9639502, -28.0313072, 12.0650434, -39.5875092, 39.5233307
40: -25.7615948, 5.3047857, -25.8494682, 5.3449612, -29.0098114, 28.9315796
41: -14.7315817, 7.3249536, -14.8222485, 7.3651781, -19.2484207, 19.2150002
42: -14.6549892, 2.1588953, -14.7402096, 2.1897924, -15.3315811, 15.3317337

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
time: 29.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
time: 32.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.8417931, 14.4816446, -14.8212376, 14.5140343, -29.3558273, 29.3028831
1: -8.3147507, 15.3998470, -8.3007765, 15.4207592, -23.4758301, 23.4451218
2: -7.1134024, 13.7947865, -7.0896816, 13.8281288, -19.5896988, 19.5744781
3: -10.5326319, 12.5097046, -10.5049763, 12.5363026, -21.3283463, 21.3185730
4: -8.0687466, 14.9157085, -8.0502071, 14.9584522, -20.8959808, 20.9172516
5: -10.6428423, 14.5073967, -10.6156425, 14.5433216, -23.5840607, 23.5408936
6: -18.7392731, 2.6683831, -18.7572021, 2.6651230, -18.9540939, 18.7857552
7: -13.6027470, 13.9203377, -13.5879669, 13.9439621, -25.6880493, 25.6498718
8: -15.1403313, 12.5055389, -15.1208286, 12.5297403, -26.2240372, 26.2012329
9: -9.7999973, 11.9111938, -9.8068943, 11.9255934, -21.3719635, 21.3447800
10: -21.2147636, 8.8269320, -21.2131824, 8.8001938, -28.7178040, 28.7281876
11: -22.3608437, 4.9461317, -22.3836536, 4.8973656, -24.6213455, 24.8098526
12: -22.5241432, 5.3585076, -22.5236626, 5.3206673, -25.7054977, 25.6340256
13: -17.6350269, 15.5735588, -17.6124687, 15.6263247, -32.7136230, 32.6261444
14: -39.4464302, -7.1341887, -39.4372711, -7.1250505, -26.8111343, 26.8310089
15: -11.9901857, 11.4615679, -11.9773464, 11.4820309, -20.9905396, 21.0855675
16: -20.1193790, 8.1809769, -20.1301346, 8.1871786, -28.3065567, 28.3111115
17: -40.6442032, -1.7618923, -40.6408844, -1.7513103, -38.8806305, 38.8789902
18: -16.6569309, 11.9876137, -16.6626835, 11.9638777, -28.6208076, 28.6502972
19: -15.8999977, 3.8551865, -15.9234047, 3.8296385, -18.8473816, 18.8967400
20: -13.4131689, 4.4887357, -13.4232502, 4.4724612, -17.8856297, 17.9119854
21: -19.0248795, 4.2580080, -19.0499306, 4.2234731, -22.6408615, 22.7620163
22: -15.6224365, 6.8031225, -15.6329823, 6.7883153, -21.8880768, 21.9997177
23: -15.0841255, 5.9787760, -15.1205196, 5.9581041, -19.7907410, 19.8710785
24: -11.5825291, 8.5704031, -11.6063576, 8.5618362, -18.5577240, 18.6192627
25: -13.9088497, 8.4253616, -13.9232178, 8.4116230, -21.3865738, 21.4398499
26: -23.3573227, 7.7372255, -23.3723240, 7.6942616, -31.0515842, 31.1095505
27: -14.6785316, 8.6953421, -14.7164249, 8.6889820, -23.3305359, 23.3813934
28: -15.3900452, 7.8387976, -15.4301453, 7.8275032, -21.2080994, 21.2675629
29: -17.5071983, 4.4119282, -17.5283012, 4.3826799, -21.0534515, 21.1012650
30: -19.3973541, 4.7264748, -19.4268093, 4.6989369, -20.9593964, 21.0317764
31: -17.0781002, 5.7699695, -17.0953712, 5.7482076, -22.8263073, 22.8653412
32: -14.5201321, 6.3027601, -14.5432301, 6.3020630, -19.4485626, 19.4250488
33: -24.2567902, 10.9035769, -24.2358894, 10.9044561, -32.8299255, 32.7980118
34: -23.5151787, 4.3434038, -23.5270443, 4.3412285, -23.3276825, 23.3361969
35: -19.6123409, 9.0733232, -19.6007996, 9.0722561, -26.6372986, 26.5831985
36: -18.4810524, 11.1458387, -18.4818783, 11.1461926, -27.3350449, 27.3365707
37: -29.1560993, 6.9249563, -29.1815205, 6.9202919, -32.8631287, 32.7321930
38: -22.8277168, 14.2547283, -22.8245888, 14.2577486, -34.6776123, 34.6454315
39: -28.0431099, 11.9990215, -28.0227585, 12.0005074, -39.5666809, 39.5424957
40: -25.8103867, 5.3455191, -25.8218765, 5.3426199, -29.0578766, 28.9381409
41: -14.7701378, 7.3578405, -14.8084393, 7.3524208, -19.2810593, 19.2134285
42: -14.6780834, 2.1924446, -14.6968174, 2.1824882, -15.3311081, 15.2572670

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4789036
time: 31.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5226506, upper bound: 10.4789036
time: 29.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.8436489, 14.4817352, -14.8686333, 14.5901794, -29.4338284, 29.3503685
1: -8.3157959, 15.3999109, -8.3309317, 15.4751081, -23.5317383, 23.4755402
2: -7.1152964, 13.7945194, -7.1260338, 13.9324360, -19.6957474, 19.6079178
3: -10.5350113, 12.5090561, -10.5443926, 12.6593370, -21.4537048, 21.3558502
4: -8.0696697, 14.9156055, -8.0876045, 15.0481148, -20.9879837, 20.9481049
5: -10.6452646, 14.5067635, -10.6581097, 14.6679344, -23.7105865, 23.5816498
6: -18.7392368, 2.6611490, -18.7764130, 2.6718864, -18.9644318, 18.8324013
7: -13.6040554, 13.9197617, -13.6236315, 13.9962711, -25.7666168, 25.6732254
8: -15.1419258, 12.5053444, -15.1512527, 12.6296101, -26.3478317, 26.2234116
9: -9.7992764, 11.9098530, -9.8403196, 11.9439735, -21.3880997, 21.3820724
10: -21.2148495, 8.8302727, -21.4170723, 8.8745575, -28.7874069, 28.9358597
11: -22.3607826, 4.9510365, -22.5972710, 4.9615974, -24.6800385, 25.0292740
12: -22.5237522, 5.3623343, -22.6993103, 5.3846879, -25.7677002, 25.8136597
13: -17.6243935, 15.5738506, -17.6312294, 15.7408848, -32.8261795, 32.6486053
14: -39.4463882, -7.1328020, -39.5968971, -7.0837355, -26.8557434, 26.9992981
15: -11.9734898, 11.4616156, -11.9789629, 11.5330296, -21.0377731, 21.1043854
16: -20.1190414, 8.1656160, -20.2216854, 8.1927700, -28.3118114, 28.3873024
17: -40.6440735, -1.7568073, -40.8369370, -1.6867371, -38.9449768, 39.0801315
18: -16.6568909, 11.9903679, -16.8277550, 12.0017633, -28.6586533, 28.8181229
19: -15.8999567, 3.8574362, -16.0581913, 3.8598218, -18.8786545, 19.0339241
20: -13.4134693, 4.4899082, -13.5239191, 4.4965239, -17.9099922, 18.0138283
21: -19.0251122, 4.2612853, -19.2217140, 4.2660112, -22.6824646, 22.9377899
22: -15.6218767, 6.8047991, -15.7116127, 6.8154135, -21.9034729, 22.1171722
23: -15.0842447, 5.9802704, -15.2214165, 5.9880991, -19.8198471, 19.9725952
24: -11.5820179, 8.5715742, -11.6800013, 8.5820961, -18.5703201, 18.6838074
25: -13.9081755, 8.4264908, -13.9994078, 8.4356918, -21.4108047, 21.5185699
26: -23.3567200, 7.7418537, -23.5712776, 7.7533774, -31.1100979, 31.3131313
27: -14.6785011, 8.6905432, -14.7776833, 8.6939096, -23.3411484, 23.4511719
28: -15.3901339, 7.8392129, -15.5137959, 7.8485260, -21.2289047, 21.3509903
29: -17.5068684, 4.4143844, -17.6192493, 4.4194484, -21.0895233, 21.1948547
30: -19.3972397, 4.7298102, -19.5434074, 4.7470603, -21.0032883, 21.1517982
31: -17.0784206, 5.7724514, -17.2465324, 5.7807798, -22.8591995, 23.0189838
32: -14.5196743, 6.3017869, -14.5641727, 6.3191762, -19.4468689, 19.4980431
33: -24.2591228, 10.9037447, -24.2810345, 11.0188885, -32.9540253, 32.8381424
34: -23.5146580, 4.3429217, -23.5504456, 4.3959212, -23.3887634, 23.3570557
35: -19.6098709, 9.0737343, -19.6271896, 9.1694927, -26.7564087, 26.6150894
36: -18.4771919, 11.1459389, -18.5022507, 11.2109222, -27.3803329, 27.3506851
37: -29.1546745, 6.9220595, -29.2185516, 6.9328828, -32.8880768, 32.7644043
38: -22.8263588, 14.2543335, -22.8651028, 14.3404856, -34.7538910, 34.6853790
39: -28.0354042, 11.9990234, -28.0360832, 12.0801105, -39.6566467, 39.5641174
40: -25.8103027, 5.3401365, -25.8526764, 5.3598018, -29.0760345, 28.9649811
41: -14.7698469, 7.3574514, -14.8246212, 7.3790731, -19.3011246, 19.2447891
42: -14.6782436, 2.1833258, -14.7421398, 2.1997859, -15.3588829, 15.3524971

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
time: 31.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4994648
time: 35.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.7884283, 14.4410629, -14.7328587, 14.4392538, -29.2276821, 29.1739216
1: -8.2666159, 15.3611937, -8.2371025, 15.3588943, -23.0079117, 23.3364029
2: -7.1118279, 13.7956085, -7.0394306, 13.7923584, -19.5954208, 20.2616959
3: -10.5125875, 12.4906197, -10.4559994, 12.4860611, -21.3097763, 22.4496307
4: -8.0560188, 14.9188480, -7.9791961, 14.9181252, -20.9269028, 21.9809875
5: -10.6235447, 14.4891586, -10.5611162, 14.4831610, -23.5264664, 24.6449890
6: -18.7579613, 2.6878538, -18.7510471, 2.6290579, -18.9214554, 18.7884064
7: -13.5470963, 13.8807812, -13.5156994, 13.8768845, -25.5688553, 26.0233078
8: -15.1027985, 12.4865360, -15.0492229, 12.4816189, -26.1665192, 26.6036682
9: -9.7491550, 11.8743219, -9.7377682, 11.8642435, -20.9274902, 21.2667084
10: -21.1612816, 8.7974081, -21.1459579, 8.7443295, -28.6202774, 28.6627045
11: -22.3693504, 4.9539461, -22.3621464, 4.8834591, -24.9344254, 24.7016754
12: -22.5174866, 5.3465357, -22.5095329, 5.2744422, -26.0354385, 25.7146072
13: -17.6085663, 15.5650330, -17.5535049, 15.5549345, -32.5770340, 33.1185379
14: -39.4477615, -7.1232166, -39.3776169, -7.1589622, -26.8180923, 26.8780746
15: -11.9923096, 11.4845390, -11.9385023, 11.4812202, -21.1447144, 21.1713943
16: -20.0719566, 8.1393299, -20.0613785, 8.1088715, -28.1808281, 28.2007084
17: -40.6531105, -1.7551556, -40.5717087, -1.8073006, -38.8458099, 38.8165512
18: -16.6300106, 11.9596024, -16.6161537, 11.9231348, -28.5531464, 28.5757561
19: -15.8806477, 3.8300881, -15.8724613, 3.7955027, -18.7782593, 18.7580872
20: -13.3882198, 4.4700165, -13.3830662, 4.4401336, -17.8283539, 17.8530827
21: -19.0204735, 4.2542715, -19.0121517, 4.1944590, -22.6531372, 22.6311035
22: -15.5990334, 6.7736406, -15.5812197, 6.7527075, -21.9225311, 21.8515015
23: -15.0669556, 5.9526486, -15.0609760, 5.9170609, -20.3283844, 19.7489929
24: -11.5658035, 8.5463266, -11.5550213, 8.5219889, -18.8573761, 18.5363274
25: -13.8840809, 8.3902674, -13.8779449, 8.3646145, -21.4934006, 21.3234558
26: -23.3040161, 7.6825962, -23.2935047, 7.6364379, -30.9404545, 30.9761009
27: -14.6696119, 8.6773167, -14.6601706, 8.6405382, -23.2899475, 23.2877655
28: -15.3765278, 7.8107586, -15.3696079, 7.7728152, -21.8946381, 21.1703339
29: -17.4906921, 4.3897209, -17.4818344, 4.3515434, -21.1237335, 21.0364990
30: -19.4144821, 4.7365298, -19.4107075, 4.6715465, -21.3281937, 21.0144310
31: -17.0473557, 5.7348289, -17.0380192, 5.7002048, -22.7475605, 22.3248749
32: -14.5315905, 6.3090196, -14.5206356, 6.2652702, -19.4140167, 19.1981049
33: -24.2142982, 10.8669014, -24.1859779, 10.8478594, -32.7344666, 32.6932220
34: -23.4895325, 4.3030105, -23.4701328, 4.2780728, -23.4048309, 23.2430725
35: -19.5743599, 9.0200653, -19.5480099, 9.0102901, -26.5133438, 26.4864960
36: -18.4546776, 11.0976534, -18.4308014, 11.0753651, -27.2399521, 27.2014694
37: -29.1348228, 6.9174604, -29.1269112, 6.8721600, -33.3238525, 32.8318710
38: -22.7661705, 14.1807146, -22.7395134, 14.1838894, -34.8078003, 34.5181427
39: -28.0002689, 11.9674902, -27.9845715, 11.9664783, -39.4799347, 39.0784760
40: -25.8019753, 5.3516970, -25.7928829, 5.3130407, -29.0058746, 28.8553772
41: -14.7814436, 7.3723154, -14.7716789, 7.3093920, -19.5552826, 19.3047295
42: -14.6973238, 2.2152591, -14.6917095, 2.1525555, -15.9417992, 15.3888474

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4756882
time: 30.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4756882
time: 28.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.7903137, 14.4411592, -14.7802839, 14.5175982, -29.3079109, 29.2214432
1: -8.2676764, 15.3612242, -8.2671852, 15.4146395, -23.0648880, 23.3668747
2: -7.1136856, 13.7953310, -7.0763555, 13.8966713, -19.7014313, 20.2978439
3: -10.5149460, 12.4899349, -10.4961739, 12.6090803, -21.4351196, 22.4877853
4: -8.0569363, 14.9187527, -8.0174198, 15.0077820, -21.0189133, 22.0135803
5: -10.6259708, 14.4885359, -10.6045389, 14.6077700, -23.6529617, 24.6872940
6: -18.7579041, 2.6806011, -18.7710648, 2.6357985, -18.9317856, 18.8360939
7: -13.5484123, 13.8802223, -13.5518742, 13.9291735, -25.6473846, 26.0473862
8: -15.1043549, 12.4863529, -15.0807095, 12.5815058, -26.2902451, 26.6269989
9: -9.7484188, 11.8729496, -9.7711573, 11.8837414, -20.9448166, 21.3040771
10: -21.1613159, 8.8008003, -21.3499680, 8.8186779, -28.6898575, 28.8705139
11: -22.3693047, 4.9588490, -22.5758266, 4.9488564, -24.9946213, 24.9209824
12: -22.5170898, 5.3503761, -22.6852379, 5.3397455, -26.0989685, 25.8943176
13: -17.5979424, 15.5653057, -17.5724411, 15.6695070, -32.6896362, 33.1377487
14: -39.4477196, -7.1218052, -39.5375099, -7.1176605, -26.8626938, 27.0460587
15: -11.9756136, 11.4845591, -11.9404135, 11.5322227, -21.1918945, 21.1906586
16: -20.0716133, 8.1239738, -20.1529980, 8.1144505, -28.1860638, 28.2769718
17: -40.6529961, -1.7501431, -40.7690430, -1.7427464, -38.9102478, 39.0188980
18: -16.6299782, 11.9623394, -16.7812958, 11.9611750, -28.5911522, 28.7436352
19: -15.8806076, 3.8323584, -16.0075054, 3.8257108, -18.8095856, 18.8955727
20: -13.3885374, 4.4711571, -13.4837914, 4.4642639, -17.8528023, 17.9549484
21: -19.0207024, 4.2575650, -19.1839485, 4.2371173, -22.6948624, 22.8069305
22: -15.5984745, 6.7753425, -15.6602783, 6.7798071, -21.9379196, 21.9688034
23: -15.0670881, 5.9541492, -15.1619253, 5.9476290, -20.3568954, 19.8505554
24: -11.5653000, 8.5475044, -11.6286592, 8.5424824, -18.8702927, 18.6009064
25: -13.8834019, 8.3914118, -13.9541721, 8.3889580, -21.5180588, 21.4021072
26: -23.3034306, 7.6871891, -23.4925308, 7.6956825, -30.9991131, 31.1797199
27: -14.6695900, 8.6725254, -14.7214355, 8.6454954, -23.3005981, 23.3575287
28: -15.3766041, 7.8111730, -15.4532852, 7.7944651, -21.9142303, 21.2537918
29: -17.4903603, 4.3921919, -17.5727997, 4.3892250, -21.1607590, 21.1301651
30: -19.4144192, 4.7398376, -19.5273323, 4.7203159, -21.3738251, 21.1344681
31: -17.0476475, 5.7373009, -17.1917419, 5.7327642, -22.7804108, 22.4816360
32: -14.5311203, 6.3080115, -14.5423374, 6.2824140, -19.4123154, 19.2722511
33: -24.2166710, 10.8670044, -24.2316933, 10.9622726, -32.8585663, 32.7336349
34: -23.4890251, 4.3025050, -23.4935875, 4.3329554, -23.4654694, 23.2639389
35: -19.5718765, 9.0204592, -19.5744476, 9.1075840, -26.6323547, 26.5184250
36: -18.4508266, 11.0977497, -18.4514198, 11.1400776, -27.2852631, 27.2156677
37: -29.1333313, 6.9145784, -29.1638927, 6.8860855, -33.3505402, 32.8640213
38: -22.7648125, 14.1802893, -22.7800293, 14.2671003, -34.8840942, 34.5580673
39: -27.9925346, 11.9674826, -28.0005646, 12.0460567, -39.5699158, 39.1021118
40: -25.8018456, 5.3462949, -25.8243179, 5.3301988, -29.0240097, 28.8828812
41: -14.7811527, 7.3719168, -14.7878056, 7.3366146, -19.5758820, 19.3360748
42: -14.6974850, 2.2061124, -14.7370672, 2.1706445, -15.9706497, 15.4840469

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 525

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4962396
time: 44.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4962396
time: 33.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 80.15 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4525027
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4525027
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4730571
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4730571
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4545472
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.5226207, upper bound: 10.4545472
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4751287
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766116, upper bound: 10.4751287
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4768374
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4768374
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4789036
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.5226506, upper bound: 10.4789036
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4973692
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766807, upper bound: 10.4994648
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4756882
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4756882
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4962396
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 80.15
Output dim: 4, lower bound: -10.4766353, upper bound: 10.4962396
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5254364, upper bound: 10.4812535
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5011276
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5030604, upper bound: 10.5035489
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5056084
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 80.15
Output dim: 4, lower bound: -10.5031186, upper bound: 10.5233643

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 66.09 + 1782.85 = 1848.94 seconds
