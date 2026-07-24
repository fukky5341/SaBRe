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
execution time: IAR + RelationalAnalysis = 2.80 + 63.45 = 66.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -10.5314132, upper bound: 10.5314132

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5087889, upper bound: 10.5287665
time: 30.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5087889, upper bound: 10.5087889
time: 30.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 61.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 61.11
Output dim: 4, lower bound: -10.5087889, upper bound: 10.5287665
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 61.11
Output dim: 4, lower bound: -10.5087889, upper bound: 10.5087889

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0786133, 23.0786285
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3593903, 20.3597641
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5543365, 22.5602570
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0912170, 22.0915146
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7565002, 24.7602463
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7954330, 18.7974167
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1445389, 26.1447678
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7308502, 26.7271042
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0118713, 21.0119171
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7734680, 28.7674866
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0322037, 25.0126724
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1541824, 26.1540833
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9943466, 26.9714661
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2403946, 21.2413712
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8148041, 18.8058281
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7595901, 22.7470856
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9351425, 21.9336624
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4196167, 20.4123230
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9373398, 18.9313354
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5904999, 21.5879745
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3831482
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9911728, 21.9866104
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2188797, 21.2160950
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4150391, 21.4014854
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4142532, 22.4061737
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2249146, 19.2329941
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7790833, 32.7875748
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4894791, 23.4936142
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5996170, 26.5998688
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2886353, 27.2933807
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4158020, 33.4124832
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9044800, 34.9105835
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1587677, 39.1587067
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8879089, 28.8880234
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6326599, 19.6369095
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0270004, 16.0305328

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4886853, upper bound: 10.5259438
time: 32.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5052771, upper bound: 10.5022276
time: 31.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0786285, 23.0786133
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3597641, 20.3593979
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5602570, 22.5543365
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0915146, 22.0912170
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7602386, 24.7565002
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7974167, 18.7954292
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1447678, 26.1445312
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7271118, 26.7308502
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0119171, 21.0118713
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7674866, 28.7734680
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0126724, 25.0322037
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1540833, 26.1541824
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9714661, 26.9943466
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2413712, 21.2403908
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8058319, 18.8148117
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7470856, 22.7595901
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9336548, 21.9351425
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4123230, 20.4196167
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9313354, 18.9373398
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5879745, 21.5904999
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3831482, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9866104, 21.9911728
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160950, 21.2188797
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4014816, 21.4150391
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4061737, 22.4142532
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2329865, 19.2249184
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7875671, 32.7790909
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4936142, 23.4894791
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5998764, 26.5996170
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2933807, 27.2886276
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4124756, 33.4158096
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9105835, 34.9044800
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1587067, 39.1587677
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8880310, 28.8879089
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6369171, 19.6326561
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0305328, 16.0270004

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5022276, upper bound: 10.5052771
time: 31.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5259438, upper bound: 10.4886853
time: 51.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 85.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 85.13
Output dim: 4, lower bound: -10.4886853, upper bound: 10.5259438
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 85.13
Output dim: 4, lower bound: -10.5052771, upper bound: 10.5022276
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 85.13
Output dim: 4, lower bound: -10.5022276, upper bound: 10.5052771
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 85.13
Output dim: 4, lower bound: -10.5259438, upper bound: 10.4886853

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0791550, 23.0785141
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3592606, 20.3596725
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5522995, 22.5598145
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0878372, 22.0901642
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7556000, 24.7599182
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7940598, 18.7957649
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1469116, 26.1441803
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7340393, 26.7268143
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0111465, 21.0097427
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7726440, 28.7597580
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0306778, 24.9983826
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1529617, 26.1518936
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9950790, 26.9611206
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2359543, 21.2419815
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8147583, 18.8011208
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7586136, 22.7378082
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9340134, 21.9372711
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4204636, 20.4094391
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9373322, 18.9312286
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5903549, 21.5878754
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3825302
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9911880, 21.9864731
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2188263, 21.2160263
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4142227, 21.3973312
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4147873, 22.4024200
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2248306, 19.2333145
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7719727, 32.7876740
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4825897, 23.4943390
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5947342, 26.5983047
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2865753, 27.2943268
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4158325, 33.4121246
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9029388, 34.9107513
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1587524, 39.1586761
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8869781, 28.8871384
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6321030, 19.6364555
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0245590, 16.0276413

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4630370, upper bound: 10.5248498
time: 31.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4876161, upper bound: 10.5002749
time: 38.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0784988, 23.0791626
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3592834, 20.3596268
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5538940, 22.5582123
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0894165, 22.0881348
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7560883, 24.7593460
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7937775, 18.7957420
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1439514, 26.1465225
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7305603, 26.7302780
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0097046, 21.0111237
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7657318, 28.7666626
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0179062, 25.0111542
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1519928, 26.1528168
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9840012, 26.9721985
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2398911, 21.2369385
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8101044, 18.8057709
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7503128, 22.7461090
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9380493, 21.9325256
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4167404, 20.4131622
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9372253, 18.9313278
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5903168, 21.5878372
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3828888
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9910355, 21.9866257
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187729, 21.2160416
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4108887, 21.4006691
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4105072, 22.4067001
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2252426, 19.2329025
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7791901, 32.7804489
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4899597, 23.4867249
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5975571, 26.5949860
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2895813, 27.2913284
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4154510, 33.4124680
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9046478, 34.9090424
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1587372, 39.1587067
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8870239, 28.8871002
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6321945, 19.6363411
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0241089, 16.0275726

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4796323, upper bound: 10.5011503
time: 30.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5042091, upper bound: 10.4765730
time: 39.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0791626, 23.0784988
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3596268, 20.3592834
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5582123, 22.5538940
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0881348, 22.0894165
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7593384, 24.7561035
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7957458, 18.7937813
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1465149, 26.1439438
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7302856, 26.7305603
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0111237, 21.0097046
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7666626, 28.7657318
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0111542, 25.0179062
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1528244, 26.1519928
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9721985, 26.9840012
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2369385, 21.2398911
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8057709, 18.8101044
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7461090, 22.7503128
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9325256, 21.9380493
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4131622, 20.4167404
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9313278, 18.9372253
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5878372, 21.5903168
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3828888, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9866257, 21.9910355
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160416, 21.2187729
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4006729, 21.4108887
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4067078, 22.4104996
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2329025, 19.2252388
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7804565, 32.7791901
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4867249, 23.4899597
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5949936, 26.5975571
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2913208, 27.2895737
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4124603, 33.4154510
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9090424, 34.9046478
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1586914, 39.1587372
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8871002, 28.8870239
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6363449, 19.6322021
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0275726, 16.0241089

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4765730, upper bound: 10.5042091
time: 39.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4876161, upper bound: 10.4796323
time: 34.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0785141, 23.0791550
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3596725, 20.3592606
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5598145, 22.5522995
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0901642, 22.0878372
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7599182, 24.7555923
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7957687, 18.7940598
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1441803, 26.1469193
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7268066, 26.7340317
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0097427, 21.0111465
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7597504, 28.7726440
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -24.9983826, 25.0306778
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1518936, 26.1529617
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9611206, 26.9950790
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2419815, 21.2359581
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8011169, 18.8147545
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7378082, 22.7586060
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9372711, 21.9340134
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4094391, 20.4204636
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9312286, 18.9373322
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5878677, 21.5903549
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3825378, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9864731, 21.9911880
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160263, 21.2188263
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.3973312, 21.4142265
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4024277, 22.4147873
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2333145, 19.2248268
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7876740, 32.7719650
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4943390, 23.4825897
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5983047, 26.5947342
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2943268, 27.2865753
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4121246, 33.4158401
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9107513, 34.9029388
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1586761, 39.1587677
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8871307, 28.8869781
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6364517, 19.6321030
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0276413, 16.0245590

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5002749, upper bound: 10.4876161
time: 39.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4876161, upper bound: 10.4630370
time: 26.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 68.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4630370, upper bound: 10.5248498
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4876161, upper bound: 10.5002749
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4796323, upper bound: 10.5011503
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.5042091, upper bound: 10.4765730
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4765730, upper bound: 10.5042091
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4876161, upper bound: 10.4796323
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.5002749, upper bound: 10.4876161
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.88
Output dim: 4, lower bound: -10.4876161, upper bound: 10.4630370

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0773315, 23.0775375
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3582458, 20.3589630
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5522156, 22.5597763
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0842514, 22.0878601
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7541809, 24.7591782
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7896194, 18.7899857
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1441956, 26.1426239
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7314453, 26.7254486
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0099106, 21.0087891
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7734985, 28.7595291
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0305557, 24.9983597
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1502991, 26.1485367
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9915695, 26.9578781
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2338638, 21.2404518
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8146439, 18.8010368
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7582397, 22.7397461
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9337769, 21.9371948
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4204102, 20.4093018
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9371796, 18.9309464
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901566, 21.5876923
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3841553
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9908066, 21.9859314
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2188187, 21.2159882
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4141693, 21.3971863
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4145966, 22.4030228
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2204781, 19.2283554
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7694397, 32.7830276
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4781265, 23.4856491
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5926208, 26.5943756
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2852325, 27.2919464
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4139252, 33.4095612
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9011383, 34.9084625
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1579437, 39.1568146
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8837891, 28.8826523
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6282959, 19.6310425
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0203514, 16.0221176

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4550014, upper bound: 10.5137105
time: 32.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4518838, upper bound: 10.5168377
time: 29.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0781708, 23.0766983
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3585510, 20.3586502
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5522614, 22.5597305
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0855408, 22.0865707
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7548370, 24.7585068
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7882843, 18.7913246
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1453552, 26.1414642
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7326660, 26.7242279
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0101929, 21.0085068
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7724152, 28.7606125
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0306549, 24.9982529
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1496048, 26.1492310
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9918365, 26.9576111
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2344284, 21.2398872
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8146667, 18.8010101
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7605515, 22.7374344
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9339371, 21.9370346
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4203262, 20.4093857
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9370575, 18.9310760
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901794, 21.5876694
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3822327
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9906464, 21.9860916
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187958, 21.2160110
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4140778, 21.3972778
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4153824, 22.4022369
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2198677, 19.2289696
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7673187, 32.7851410
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4738998, 23.4898758
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5908203, 26.5961914
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2841949, 27.2929840
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4132690, 33.4102173
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9006500, 34.9089584
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1569061, 39.1578674
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8824921, 28.8839569
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6266937, 19.6326485
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0190315, 16.0234375

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4795820, upper bound: 10.4891329
time: 35.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4764634, upper bound: 10.4922606
time: 37.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0766830, 23.0781784
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3582687, 20.3589249
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5538101, 22.5581741
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0858307, 22.0858383
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7546844, 24.7585983
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7893372, 18.7899628
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1412354, 26.1449661
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7279663, 26.7289124
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0084686, 21.0101700
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7665863, 28.7664413
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0177841, 25.0111313
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1493301, 26.1494598
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9804916, 26.9689560
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2377930, 21.2354088
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8099899, 18.8056870
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7499390, 22.7480469
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9378128, 21.9324493
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4166870, 20.4130249
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9370804, 18.9310532
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901108, 21.5876541
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3845062
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9906540, 21.9860840
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187653, 21.2160110
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4108353, 21.4005241
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4103165, 22.4073029
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2208900, 19.2279396
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7766571, 32.7758102
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4855042, 23.4780350
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5954285, 26.5910645
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2882233, 27.2889557
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4135437, 33.4099045
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9028473, 34.9067535
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1579132, 39.1568451
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8838501, 28.8826141
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6284027, 19.6309280
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0199051, 16.0220451

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4715954, upper bound: 10.4900016
time: 30.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4684765, upper bound: 10.4931222
time: 31.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0775223, 23.0773468
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3585815, 20.3586121
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5538559, 22.5581284
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0871201, 22.0845490
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7553558, 24.7579269
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7880020, 18.7912979
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1423950, 26.1438065
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7291870, 26.7276917
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0087509, 21.0098877
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7655029, 28.7675247
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0178833, 25.0110245
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1486282, 26.1501541
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9807587, 26.9686890
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2383575, 21.2348404
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8100204, 18.8056602
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7522507, 22.7457352
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9379730, 21.9322891
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4166031, 20.4131088
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9369507, 18.9311829
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901413, 21.5876312
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3825912
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9904938, 21.9862442
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187424, 21.2160339
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4107437, 21.4006195
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4111023, 22.4065170
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2202797, 19.2285538
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7745361, 32.7779160
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4812698, 23.4822693
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5936279, 26.5928726
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2871857, 27.2899933
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4128876, 33.4105530
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9023590, 34.9072495
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1568756, 39.1578827
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8825378, 28.8839188
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6267853, 19.6325340
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0185814, 16.0233688

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4961743, upper bound: 10.4654238
time: 37.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4930548, upper bound: 10.4685443
time: 33.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0773468, 23.0775223
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3586121, 20.3585815
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5581284, 22.5538559
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0845490, 22.0871201
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7579193, 24.7553558
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7912979, 18.7880020
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1437988, 26.1423874
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7276917, 26.7291946
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0098877, 21.0087509
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7675247, 28.7655029
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0110245, 25.0178833
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1501541, 26.1486282
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9686890, 26.9807587
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2348404, 21.2383575
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8056641, 18.8100166
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7457352, 22.7522507
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9322891, 21.9379730
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4131088, 20.4166031
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9311829, 18.9369507
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5876312, 21.5901413
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3825912, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9862442, 21.9904938
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160339, 21.2187424
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4006195, 21.4107437
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4065170, 22.4111023
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2285576, 19.2202759
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7779236, 32.7745438
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4822693, 23.4812698
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5928650, 26.5936279
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2899933, 27.2871933
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4105530, 33.4128876
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9072571, 34.9023514
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1578827, 39.1568756
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8839111, 28.8825378
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6325378, 19.6267891
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0233650, 16.0185852

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4685443, upper bound: 10.4930548
time: 36.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4654238, upper bound: 10.4961743
time: 31.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0781784, 23.0766830
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3589249, 20.3582687
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5581741, 22.5538101
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0858383, 22.0858307
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7585907, 24.7546844
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7899628, 18.7893372
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1449585, 26.1412277
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7289124, 26.7279739
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0101700, 21.0084686
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7664413, 28.7665863
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0111313, 25.0177841
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1494598, 26.1493301
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9689560, 26.9804916
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2354050, 21.2377930
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8056870, 18.8099937
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7480469, 22.7499390
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9324493, 21.9378128
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4130249, 20.4166870
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9310532, 18.9370804
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5876541, 21.5901108
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3845062, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9860840, 21.9906540
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160110, 21.2187653
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4005203, 21.4108353
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4073029, 22.4103165
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2279396, 19.2208939
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7758026, 32.7766571
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4780350, 23.4855042
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5910645, 26.5954361
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2889557, 27.2882309
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4098969, 33.4135437
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9067535, 34.9028549
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1568451, 39.1579285
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8826141, 28.8838425
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6309204, 19.6283989
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0220451, 16.0199051

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4684765
time: 33.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4900016, upper bound: 10.4715954
time: 46.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0766983, 23.0781708
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3586502, 20.3585510
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5597305, 22.5522614
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0865707, 22.0855408
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7584991, 24.7548447
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7913208, 18.7882805
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1414642, 26.1453552
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7242279, 26.7326660
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0085068, 21.0101929
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7606125, 28.7724152
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -24.9982529, 25.0306549
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1492310, 26.1496048
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9576111, 26.9918365
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2398911, 21.2344284
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8010101, 18.8146706
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7374344, 22.7605515
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9370346, 21.9339371
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4093857, 20.4203262
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9310760, 18.9370575
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5876694, 21.5901794
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3822327, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9860916, 21.9906464
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160110, 21.2187958
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.3972778, 21.4140778
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4022369, 22.4153824
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2289696, 19.2198639
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7851410, 32.7673264
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4898758, 23.4738998
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5961914, 26.5908051
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2929840, 27.2842026
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4102173, 33.4132767
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9089661, 34.9006424
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1578522, 39.1569061
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8839569, 28.8824921
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6326447, 19.6266899
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0234375, 16.0190315

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4922607, upper bound: 10.4764634
time: 37.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4891329, upper bound: 10.4795820
time: 33.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0775375, 23.0773315
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3589630, 20.3582458
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5597763, 22.5522156
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0878601, 22.0842514
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7591705, 24.7541809
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7899857, 18.7896194
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1426239, 26.1441956
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7254486, 26.7314453
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0087891, 21.0099106
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7595291, 28.7734985
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -24.9983597, 25.0305557
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1485291, 26.1502991
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9578781, 26.9915695
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2404556, 21.2338638
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8010330, 18.8146439
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7397461, 22.7582397
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9371948, 21.9337769
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4093018, 20.4204102
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9309540, 18.9371796
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5876923, 21.5901566
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3841553, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9859314, 21.9908066
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2159882, 21.2188187
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.3971863, 21.4141731
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4030228, 22.4145966
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2283516, 19.2204781
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7830353, 32.7694321
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4856491, 23.4781265
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5943756, 26.5926208
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2919464, 27.2852402
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4095612, 33.4139252
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9084625, 34.9011459
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1568146, 39.1579437
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8826599, 28.8837967
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6310425, 19.6282997
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0221138, 16.0203552

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4518838
time: 33.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.5137105, upper bound: 10.4550014
time: 55.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 92.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4550014, upper bound: 10.5137105
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4518838, upper bound: 10.5168377
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4795820, upper bound: 10.4891329
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4764634, upper bound: 10.4922606
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4715954, upper bound: 10.4900016
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4684765, upper bound: 10.4931222
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4961743, upper bound: 10.4654238
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4930548, upper bound: 10.4685443
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4685443, upper bound: 10.4930548
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4654238, upper bound: 10.4961743
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4684765
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4900016, upper bound: 10.4715954
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4922607, upper bound: 10.4764634
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4891329, upper bound: 10.4795820
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4518838
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 92.12
Output dim: 4, lower bound: -10.5137105, upper bound: 10.4550014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0774841, 23.0774612
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3586731, 20.3587646
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5531387, 22.5593796
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0842590, 22.0878372
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7543259, 24.7591095
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7893906, 18.7897224
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1441956, 26.1426392
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7315826, 26.7252502
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0099106, 21.0087967
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7733459, 28.7593002
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0303116, 24.9987488
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1504211, 26.1483002
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9910812, 26.9578171
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2345428, 21.2402115
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8142090, 18.8017731
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7578278, 22.7405014
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9336853, 21.9372253
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4202271, 20.4095306
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9369583, 18.9313660
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5897217, 21.5886307
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3843994
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9906158, 21.9862061
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2188034, 21.2159805
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4138794, 21.3974648
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4141769, 22.4036942
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2218018, 19.2279701
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7696228, 32.7829742
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4794922, 23.4854965
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5926361, 26.5943832
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2850952, 27.2918472
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4132843, 33.4110184
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9012299, 34.9084091
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1577759, 39.1574402
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8836288, 28.8825531
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6284180, 19.6308975
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0206261, 16.0219917

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4537844, upper bound: 10.4892611
time: 37.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4274871, upper bound: 10.5124389
time: 36.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0772629, 23.0775375
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3580475, 20.3589630
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5518188, 22.5597763
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0842285, 22.0878601
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7541122, 24.7591782
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7893600, 18.7899857
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1441956, 26.1426239
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7312317, 26.7254486
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0099106, 21.0087891
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7732697, 28.7595291
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0305557, 24.9981232
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1500702, 26.1485367
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9915085, 26.9578781
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2336197, 21.2404518
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8146439, 18.8005905
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7582397, 22.7393417
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9337769, 21.9371033
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4204102, 20.4091263
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9371796, 18.9307251
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901566, 21.5872574
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3839340
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9908066, 21.9857407
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2188110, 21.2159882
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4141693, 21.3968964
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4145966, 22.4026031
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2200928, 19.2283554
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7693787, 32.7830276
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4779739, 23.4856491
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5926361, 26.5943756
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2852325, 27.2918091
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4139252, 33.4089203
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9010925, 34.9084625
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1579437, 39.1566620
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8836899, 28.8826523
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6281586, 19.6310425
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0202293, 16.0221176

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4274871, upper bound: 10.4923853
time: 31.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4274871, upper bound: 10.5155666
time: 38.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0783234, 23.0766220
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3589859, 20.3584595
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5531845, 22.5593338
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0855484, 22.0865479
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7549973, 24.7584381
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7880554, 18.7910614
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1453552, 26.1414719
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7328033, 26.7240295
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0101929, 21.0085144
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7722626, 28.7603836
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0304184, 24.9986496
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1497269, 26.1490021
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9913483, 26.9575500
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2351074, 21.2396469
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8142242, 18.8017502
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7601395, 22.7381897
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9338455, 21.9370651
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4201431, 20.4096146
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9368286, 18.9314957
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5897446, 21.5886078
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3824768
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9904556, 21.9863663
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187805, 21.2160034
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4137878, 21.3975601
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4149628, 22.4029160
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2211838, 19.2285843
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7675018, 32.7850876
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4752579, 23.4897232
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5908203, 26.5961914
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2840576, 27.2928848
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4126434, 33.4116745
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9007263, 34.9089127
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1567383, 39.1584778
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8823166, 28.8838577
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6268005, 19.6325035
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0193062, 16.0233116

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4783655, upper bound: 10.4646783
time: 33.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4551867, upper bound: 10.4878593
time: 35.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0780945, 23.0766983
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3583603, 20.3586502
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5518646, 22.5597305
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0855179, 22.0865707
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7547684, 24.7585068
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7880173, 18.7913246
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1453552, 26.1414642
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7324524, 26.7242279
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0101929, 21.0085068
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7721863, 28.7606125
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0306549, 24.9980164
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1493683, 26.1492310
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9917679, 26.9576111
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2341843, 21.2398872
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8146667, 18.8005676
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7605515, 22.7370300
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9339371, 21.9369431
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4203262, 20.4092026
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9370575, 18.9308548
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901794, 21.5872345
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3820114
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9906464, 21.9859009
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187805, 21.2160110
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4140778, 21.3969879
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4153824, 22.4018173
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2194824, 19.2289696
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7672729, 32.7851410
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4737473, 23.4898758
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5908051, 26.5961914
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2841949, 27.2928467
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4132690, 33.4095764
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9005890, 34.9089584
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1569061, 39.1576996
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8823929, 28.8839569
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6265411, 19.6326485
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0189095, 16.0234375

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4752475, upper bound: 10.4678024
time: 32.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4274871, upper bound: 10.4909869
time: 34.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0768280, 23.0781097
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3587036, 20.3587265
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5547409, 22.5577850
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0858383, 22.0858154
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7548294, 24.7585297
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7891083, 18.7896996
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1412354, 26.1449814
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7281189, 26.7287140
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0084686, 21.0101776
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7664337, 28.7662048
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0175400, 25.0115204
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1494522, 26.1492310
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9800034, 26.9688873
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2384720, 21.2351685
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8095551, 18.8064270
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7495346, 22.7488022
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9377213, 21.9324799
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4165039, 20.4132462
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9368515, 18.9314728
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5896759, 21.5885925
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3847580
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9904556, 21.9863586
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187500, 21.2160034
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4105453, 21.4008026
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4098969, 22.4079742
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2222137, 19.2275543
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7768402, 32.7757492
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4868622, 23.4778824
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5954437, 26.5910645
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2880859, 27.2888565
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4129028, 33.4113617
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9029388, 34.9067001
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1577606, 39.1574554
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8836746, 28.8825150
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6285095, 19.6307831
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0201759, 16.0219231

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4703895, upper bound: 10.4656150
time: 51.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4472017, upper bound: 10.4887835
time: 37.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0766068, 23.0781784
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3580704, 20.3589249
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5534210, 22.5581741
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0858078, 22.0858383
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7546158, 24.7585983
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7890778, 18.7899628
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1412354, 26.1449738
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7277832, 26.7289124
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0084686, 21.0101624
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7663574, 28.7664413
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0177841, 25.0108948
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1491013, 26.1494598
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9804306, 26.9689560
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2375565, 21.2354088
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8099899, 18.8052444
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7499390, 22.7476349
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9378128, 21.9323578
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4166870, 20.4128418
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9370804, 18.9308319
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901108, 21.5872269
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3842926
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9906540, 21.9858932
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187500, 21.2160110
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4108353, 21.4002304
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4103165, 22.4068832
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2205048, 19.2279396
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7765961, 32.7758102
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4853439, 23.4780350
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5954437, 26.5910645
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2882233, 27.2888184
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4135437, 33.4092636
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9028015, 34.9067535
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1579132, 39.1566772
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8837509, 28.8826141
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6282501, 19.6309280
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0197792, 16.0220451

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4672708, upper bound: 10.4687304
time: 30.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4440872, upper bound: 10.4919040
time: 34.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0776672, 23.0772705
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3590164, 20.3584137
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5547867, 22.5577393
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0871277, 22.0845261
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7555008, 24.7578583
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7877731, 18.7910385
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1423950, 26.1438217
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7293396, 26.7274933
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0087509, 21.0098877
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7653503, 28.7672882
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0176468, 25.0114212
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1487503, 26.1499252
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9802704, 26.9686279
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2390366, 21.2346039
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8095703, 18.8064003
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7518463, 22.7464905
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9378815, 21.9323196
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4164200, 20.4133301
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9367294, 18.9315948
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5897064, 21.5885696
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3828354
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9903030, 21.9865189
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187271, 21.2160263
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4104538, 21.4008980
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4106827, 22.4071960
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2216034, 19.2281723
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7747192, 32.7778625
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4826355, 23.4821091
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5936432, 26.5928802
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2870636, 27.2898865
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4122467, 33.4120178
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9024353, 34.9072037
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1567230, 39.1585083
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8823776, 28.8838196
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6269073, 19.6323891
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0188560, 16.0232430

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4949686, upper bound: 10.4410290
time: 36.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4717896, upper bound: 10.4642049
time: 38.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0774460, 23.0773468
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3583832, 20.3586121
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5534668, 22.5581284
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0870972, 22.0845490
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7552872, 24.7579269
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7877350, 18.7912979
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1423950, 26.1438141
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7290039, 26.7276917
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0087509, 21.0098801
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7652740, 28.7675247
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0178833, 25.0107880
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1483994, 26.1501541
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9806976, 26.9686890
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2381210, 21.2348404
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8100204, 18.8052177
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7522507, 22.7453232
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9379730, 21.9321976
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4166031, 20.4129257
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9369507, 18.9309616
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5901413, 21.5871964
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3851891, 23.3823700
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9904938, 21.9860535
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2187271, 21.2160339
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4107437, 21.4003258
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4111023, 22.4060974
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2198944, 19.2285538
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7744904, 32.7779160
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4811172, 23.4822693
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5936279, 26.5928726
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2871857, 27.2898560
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4128876, 33.4099197
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9022980, 34.9072495
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1568756, 39.1577148
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8824387, 28.8839188
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6266479, 19.6325340
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0184593, 16.0233688

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4918493, upper bound: 10.4441438
time: 44.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4686749, upper bound: 10.4673251
time: 33.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8618870, 14.4510136, -14.8618870, 14.4510136, -29.3129005, 29.3129005
1: -8.3281975, 15.3678808, -8.3281975, 15.3678808, -23.0774918, 23.0774460
2: -7.1288767, 13.8021660, -7.1288767, 13.8021660, -20.3590469, 20.3583832
3: -10.5496674, 12.4976358, -10.5496674, 12.4976358, -22.5590591, 22.5534668
4: -8.0850925, 14.9252291, -8.0850925, 14.9252291, -22.0845566, 22.0870972
5: -10.6610279, 14.4965401, -10.6610279, 14.4965401, -24.7580795, 24.7552872
6: -18.7663441, 2.6774015, -18.7663441, 2.6774015, -18.7910690, 18.7877388
7: -13.6185160, 13.8870392, -13.6185160, 13.8870392, -26.1438141, 26.1424026
8: -15.1561260, 12.4972715, -15.1561260, 12.4972715, -26.7278442, 26.7289963
9: -9.8208799, 11.8768606, -9.8208799, 11.8768606, -21.0098801, 21.0087585
10: -21.2251701, 8.8218918, -21.2251701, 8.8218918, -28.7673645, 28.7652740
11: -22.3840084, 4.9667740, -22.3840084, 4.9667740, -25.0107880, 25.0182724
12: -22.5279617, 5.3786635, -22.5279617, 5.3786635, -26.1502762, 26.1483994
13: -17.6499367, 15.5782642, -17.6499367, 15.5782642, -33.2282028, 33.2282028
14: -39.4582977, -7.1018515, -39.4582977, -7.1018515, -26.9682007, 26.9806976
15: -11.9995604, 11.4915495, -11.9995604, 11.4915495, -21.2355270, 21.2381210
16: -20.1465702, 8.1426086, -20.1465702, 8.1426086, -28.2891788, 28.2891788
17: -40.6544800, -1.7256451, -40.6544800, -1.7256451, -38.9288330, 38.9288330
18: -16.6360550, 12.0172043, -16.6360550, 12.0172043, -28.6532593, 28.6532593
19: -15.8925772, 3.8685293, -15.8925772, 3.8685293, -18.8052216, 18.8107567
20: -13.4005413, 4.5008249, -13.4005413, 4.5008249, -17.9013672, 17.9013672
21: -19.0358086, 4.2749958, -19.0358086, 4.2749958, -22.7453232, 22.7530060
22: -15.6058636, 6.8217492, -15.6058636, 6.8217492, -21.9321976, 21.9380035
23: -15.0769577, 5.9918957, -15.0769577, 5.9918957, -20.4129257, 20.4168320
24: -11.5693779, 8.5922823, -11.5693779, 8.5922823, -18.9309540, 18.9373703
25: -13.8957558, 8.4426670, -13.8957558, 8.4426670, -21.5871964, 21.5910797
26: -23.3176651, 7.7645259, -23.3176651, 7.7645259, -31.0821915, 31.0821915
27: -14.6798315, 8.7053566, -14.6798315, 8.7053566, -23.3823700, 23.3851891
28: -15.3863468, 7.8504777, -15.3863468, 7.8504777, -21.9860535, 21.9907684
29: -17.5053177, 4.4264417, -17.5053177, 4.4264417, -21.2160187, 21.2187271
30: -19.4236107, 4.7405787, -19.4236107, 4.7405787, -21.4003220, 21.4110222
31: -17.0602951, 5.7935114, -17.0602951, 5.7935114, -22.4060974, 22.4117737
32: -14.5406809, 6.3118315, -14.5406809, 6.3118315, -19.2298813, 19.2198944
33: -24.2363091, 10.9148712, -24.2363091, 10.9148712, -32.7781067, 32.7744904
34: -23.5024014, 4.3531070, -23.5024014, 4.3531070, -23.4836273, 23.4811172
35: -19.5896034, 9.0823793, -19.5896034, 9.0823793, -26.5928802, 26.5936279
36: -18.4679565, 11.1555643, -18.4679565, 11.1555643, -27.2898560, 27.2870941
37: -29.1525478, 6.9361892, -29.1525478, 6.9361892, -33.4099121, 33.4143524
38: -22.7857018, 14.2640705, -22.7857018, 14.2640705, -34.9073334, 34.9023056
39: -28.0193806, 12.0090618, -28.0193806, 12.0090618, -39.1577148, 39.1575012
40: -25.8200474, 5.3555393, -25.8200474, 5.3555393, -28.8837509, 28.8824387
41: -14.7926311, 7.3673558, -14.7926311, 7.3673558, -19.6326447, 19.6266441
42: -14.7083321, 2.2029827, -14.7083321, 2.2029827, -16.0236397, 16.0184593

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4673251, upper bound: 10.4686749
time: 34.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -10.4441438, upper bound: 10.4918493
time: 39.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 77.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4537844, upper bound: 10.4892611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4274871, upper bound: 10.5124389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4274871, upper bound: 10.4923853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4274871, upper bound: 10.5155666
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4783655, upper bound: 10.4646783
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4551867, upper bound: 10.4878593
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4752475, upper bound: 10.4678024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4274871, upper bound: 10.4909869
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4703895, upper bound: 10.4656150
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4472017, upper bound: 10.4887835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4672708, upper bound: 10.4687304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4440872, upper bound: 10.4919040
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4949686, upper bound: 10.4410290
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4717896, upper bound: 10.4642049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4918493, upper bound: 10.4441438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4686749, upper bound: 10.4673251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4673251, upper bound: 10.4686749
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.19
Output dim: 4, lower bound: -10.4441438, upper bound: 10.4918493
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4654238, upper bound: 10.4961743
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4684765
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4900016, upper bound: 10.4715954
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4922607, upper bound: 10.4764634
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4891329, upper bound: 10.4795820
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.4931222, upper bound: 10.4518838
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.19
Output dim: 4, lower bound: -10.5137105, upper bound: 10.4550014

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 66.25 + 1783.11 = 1849.36 seconds
